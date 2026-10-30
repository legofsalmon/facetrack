"""People segmentation for the silhouette cutout mode.

Four engines share one interface. mask(frame, roi) returns a full-frame
uint8 alpha; tile(image) returns the alpha of one crop at the crop's own
size, which is what the pipeline's crowd scene uses to run the engine
once per tracked person. Loaded lazily the first time the 'people'
cutout shape is selected.

- PP-HumanSeg (OpenCV model zoo, Apache-2.0): 192x192 via cv2.dnn, CPU.
- Selfie (Google MediaPipe selfie segmenter, Apache-2.0): 144x256 via
  ONNX Runtime, the fastest of the four and the cleanest on small crops.
- MODNet (Apache-2.0): true portrait matte with hair detail; strongest
  on a prominent single subject; sized to the machine.
- RVM (GPL-3.0, internal builds only): recurrent video matting, full
  frame only.

Every one of them is a *portrait* model: measured on a wide shot of six
people they key one person, or the ground, or nothing, and only RVM
finds everyone. Fed one body-shaped crop per face they all recover —
that is why the pipeline has a crowd scene. mask() takes an optional
region of interest for the single scene (one body-shaped expansion of
all the faces), and the result is pasted back into a full-frame mask.
Geometry follows the OpenCV-zoo reference (squashed resize in,
probability field resized up before any thresholding) and the
confidence gate is centred on 50% so the matte doesn't erode one-sided.

Temporal smoothing lives in the pipeline (full-frame space), so it stays
correct while the regions follow moving subjects.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = _MODELS_DIR / "human_segmentation_pphumanseg_2023mar.onnx"
MODNET_PATH = _MODELS_DIR / "modnet_portrait.onnx"
SELFIE_PATH = _MODELS_DIR / "selfie_segmenter_landscape.onnx"
RVM_PATH = _MODELS_DIR / "rvm_mobilenetv3_fp32.onnx"
_INPUT = 192


def _ort_session(path: Path, threads: int | None = None):
    from .runtime import make_session
    if not path.exists():
        raise RuntimeError(f"{path.name} missing — run the launcher or "
                           "`python main.py --doctor` to download it")
    return make_session(path, threads=threads)

def _paste(infer, frame: np.ndarray, roi, min_px: int) -> np.ndarray:
    """Run infer on the ROI (or the whole frame) and return a full-frame
    mask. A degenerate ROI falls back to the whole frame."""
    H, W = frame.shape[:2]
    if roi is None:
        return infer(frame)
    x1, y1, x2, y2 = roi
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < min_px or y2 - y1 < min_px:
        return infer(frame)
    out = np.zeros((H, W), dtype=np.uint8)
    out[y1:y2, x1:x2] = infer(frame[y1:y2, x1:x2])
    return out


# 25%..75% probability -> 0..255 alpha; the 50% midpoint lands on 128 so
# the cutout's re-harden threshold decides exactly like the zoo's argmax.
_GATE_LUT = np.clip((np.arange(256, dtype=np.float32) - 0.25 * 255) * 2.0,
                    0, 255).astype(np.uint8)


class PeopleSegmenter:
    """PP-HumanSeg via cv2.dnn.

    Worth knowing, because it looks like an oversight and the decision
    was made deliberately: this is the one model that does NOT go
    through runtime.make_session, so it is CPU-only and never sees the
    CUDA provider the others get on the Windows machine. Moving it to
    ONNX Runtime measured 5.5 ms against cv2.dnn's 8.1 ms on CPU alone,
    before any GPU, so the prize is real.

    It was not done because the two backends do not agree. On a frame
    with no clear subject the logits sit near the decision boundary and
    the masks diverge across 6.8% of the frame at the 50% contour the
    cutout re-hardens on — which may be nothing on real footage with
    real people in it, or may be a visibly different silhouette. There
    is no crowd footage in the repo to tell the difference, and this is
    the feed that goes to the wall. Try it against a real clip before
    switching.
    """

    soft = False  # coarse 192px mask: downstream re-hardens the edge
    tiles = True  # can run on one crop at a time (crowd scene)
    min_px = 8
    provider = "cv2.dnn (CPU)"

    def __init__(self, model_path: str | Path = MODEL_PATH):
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                "people-silhouette model missing — run the launcher or "
                "`python main.py --doctor` to download it")
        self.net = cv2.dnn.readNet(str(path))

    def tile(self, image: np.ndarray) -> np.ndarray:
        """Alpha of one crop, at the crop's size."""
        return self._infer(image)

    def _infer(self, image: np.ndarray) -> np.ndarray:
        """uint8 person-probability (gated) at the image's own size."""
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA),
            1 / 127.5, (_INPUT, _INPUT), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        logits = self.net.forward()[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        person = (e[1] / e.sum(axis=0)).astype(np.float32)
        p8 = cv2.medianBlur((person * 255).astype(np.uint8), 3)
        full = cv2.resize(p8, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.LUT(full, _GATE_LUT)

    def mask(self, frame: np.ndarray,
             roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """Full-frame uint8 alpha (0..255) of everyone in the picture.

        roi (x1, y1, x2, y2): run the net on just that region — much
        better detail when people are known to be there."""
        return _paste(self._infer, frame, roi, self.min_px)


class SelfieSegmenter:
    """Google's MediaPipe selfie segmenter (landscape, 144x256) through
    ONNX Runtime. Apache-2.0, a quarter of a megabyte, and the fastest
    engine here: 11 ms a 1080p mask against PP-HumanSeg's 20 on the same
    4-core box, and about 4 ms a person on body-sized crops, where it was
    also the cleanest of the licence-clean engines. Built for video-call
    framing, so like the others it wants people to fill its input.

    The ONNX file is the TFLite model from Google's model store converted
    with tf2onnx, unchanged otherwise (models/README.md). Input is NHWC
    RGB in 0..1; the output is the person probability, already a sigmoid.
    """

    soft = False  # 144x256 mask: downstream re-hardens the edge
    tiles = True
    min_px = 16
    _IN_H, _IN_W = 144, 256

    def __init__(self, model_path: str | Path = SELFIE_PATH):
        from .runtime import active_provider
        # One thread: at 144x256 the model is 3 ms single-threaded and
        # more threads only take cores from the show loop.
        self.sess = _ort_session(Path(model_path), threads=1)
        self.provider = active_provider(self.sess)
        self._in = self.sess.get_inputs()[0].name

    def _infer(self, image: np.ndarray) -> np.ndarray:
        x = cv2.resize(image, (self._IN_W, self._IN_H), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255.0)
        p = self.sess.run(None, {self._in: x[None]})[0]
        p = np.asarray(p, dtype=np.float32).reshape(self._IN_H, self._IN_W)
        p8 = cv2.medianBlur((np.clip(p, 0, 1) * 255).astype(np.uint8), 3)
        full = cv2.resize(p8, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.LUT(full, _GATE_LUT)

    def tile(self, image: np.ndarray) -> np.ndarray:
        return self._infer(image)

    def mask(self, frame: np.ndarray,
             roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        return _paste(self._infer, frame, roi, self.min_px)


class ModnetMatter:
    """MODNet portrait matting (Apache-2.0) via onnxruntime: true soft
    alpha with hair-level detail. Strongest on prominent single subjects
    (a presenter to camera); GPU-recommended, workable on CPU."""

    soft = True  # a real matte: don't re-harden the edge downstream
    tiles = True
    min_px = 32

    def __init__(self, model_path: str | Path = MODNET_PATH, ref: int | None = None):
        from .runtime import active_provider
        self.sess = _ort_session(Path(model_path))
        self.provider = active_provider(self.sess)
        # Sized to the machine: 512 px on a GPU, where it is cheap; 320 on
        # a CPU, where 512 measured four masks a second at 1080p (a lag
        # an audience sees) and 256 three times that.
        if ref is None:
            ref = 512 if not self.provider.startswith("CPU") else 320
        self.ref = ref

    def tile(self, image: np.ndarray) -> np.ndarray:
        return self._infer(image)

    def _infer(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        s = self.ref / max(H, W)
        nh = max(32, int(H * s) // 32 * 32)
        nw = max(32, int(W * s) // 32 * 32)
        x = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        # transpose gives a view with the wrong strides, which ONNX
        # Runtime then has to copy internally. Do it here, where it is
        # visible, rather than paying for it invisibly every frame.
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        m = self.sess.run(None, {"input": blob})[0][0, 0]
        m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        return cv2.resize(m8, (W, H), interpolation=cv2.INTER_LINEAR)

    def mask(self, frame: np.ndarray,
             roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        return _paste(self._infer, frame, roi, self.min_px)


class RvmMatter:
    """Robust Video Matting (GPL-3.0) via onnxruntime: the best-looking
    people matte — temporally stable by design (recurrent state carries
    between frames). Runs full-frame; the ROI is ignored so the memory
    stays coherent. GPU-recommended; ~27ms/frame on an M-class CPU at
    720p, so workable for testing."""

    soft = True
    tiles = False  # recurrent: one coherent stream, never crops

    def __init__(self, model_path: str | Path = RVM_PATH):
        from .runtime import active_provider
        self.sess = _ort_session(Path(model_path))
        self.provider = active_provider(self.sess)
        self._state = None
        self._shape: tuple[int, int] | None = None

    def mask(self, frame: np.ndarray,
             roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        H, W = frame.shape[:2]
        if self._shape != (W, H) or self._state is None:
            self._shape = (W, H)
            self._state = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ds = float(np.clip(384.0 / max(W, H), 0.125, 1.0))
        inputs = {"src": x.transpose(2, 0, 1)[None],
                  "downsample_ratio": np.array([ds], np.float32)}
        for key, val in zip(("r1i", "r2i", "r3i", "r4i"), self._state):
            inputs[key] = val
        _fgr, pha, *self._state = self.sess.run(None, inputs)
        return (np.clip(pha[0, 0], 0, 1) * 255).astype(np.uint8)


# The people-cutout engines. `distributable` marks whether a model may
# ship in a sold/distributed build — RVM is GPL-3.0, so it stays in
# internal builds only (see yewee/edition.py and LICENSE).
PEOPLE_MODELS = {
    "selfie": {"cls": SelfieSegmenter, "path": SELFIE_PATH,
               "label": "Fast — Selfie (MediaPipe)", "licence": "Apache-2.0",
               "distributable": True},
    "pphumanseg": {"cls": PeopleSegmenter, "path": MODEL_PATH,
                   "label": "Fast — PP-HumanSeg", "licence": "Apache-2.0",
                   "distributable": True},
    "modnet": {"cls": ModnetMatter, "path": MODNET_PATH,
               "label": "Quality — MODNet", "licence": "Apache-2.0",
               "distributable": True},
    "rvm": {"cls": RvmMatter, "path": RVM_PATH,
            "label": "Best — RVM video matting", "licence": "GPL-3.0",
            "distributable": False},
}

#: The model the pipeline falls back to when the chosen one cannot load.
DEFAULT_PEOPLE_MODEL = "pphumanseg"

#: The panel's "Auto" entry: not a model, a rule (resolve_people_model).
AUTO_LABEL = "Auto — picked for this machine and scene"


def resolve_people_model(name: str, scene: str, provider: str) -> str:
    """Turn the panel's choice into a concrete engine.

    'auto' reads the machine and the scene. With a GPU provider loaded
    (CUDA, DirectML, CoreML) a single subject gets MODNet's real matte,
    and a crowd gets the Selfie model on one crop per person — small
    crops, where its 144x256 input is plenty and its speed lets the
    tile count grow. On a CPU both scenes get Selfie: MODNet on a CPU is
    a few masks a second at 1080p, which an audience sees as lag.
    Anything 'auto' would pick that this build cannot load falls to the
    next choice, so a missing file never breaks the cutout.
    """
    if name != "auto":
        return name
    gpu = not provider.startswith("CPU")
    order = (["modnet", "selfie", "pphumanseg"] if gpu and scene == "single"
             else ["selfie", "pphumanseg", "modnet"])
    for key in order:
        if _usable(key):
            return key
    return DEFAULT_PEOPLE_MODEL


def _usable(key: str) -> bool:
    """A model is offered only if it may ship in this build AND is present."""
    from .edition import DISTRIBUTION
    info = PEOPLE_MODELS.get(key)
    if info is None or (DISTRIBUTION and not info["distributable"]):
        return False
    return Path(info["path"]).exists()


def available_people_models() -> list[dict]:
    """[{value, label}] for the panel — Auto, then what this build can
    actually run."""
    return [{"value": "auto", "label": AUTO_LABEL}] + [
        {"value": k, "label": PEOPLE_MODELS[k]["label"]}
        for k in PEOPLE_MODELS if _usable(k)]


def create_people_model(name: str):
    """Instantiate a people model, refusing anything this build may not
    ship (the caller falls back and reports it in the panel)."""
    if not _usable(name):
        info = PEOPLE_MODELS.get(name)
        if info is not None and not info["distributable"]:
            raise RuntimeError(
                f"{info['label']} ({info['licence']}) is not included in this build")
        raise RuntimeError(f"people model '{name}' is not available")
    return PEOPLE_MODELS[name]["cls"]()
