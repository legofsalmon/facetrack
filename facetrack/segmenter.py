"""People segmentation for the silhouette cutout mode.

PP-HumanSeg (OpenCV model zoo, Apache-2.0): a matte of every person in
the picture, 192x192 input via cv2.dnn — CPU-friendly, so it works on
both the Mac and the show machine. Loaded lazily the first time the
'people' cutout shape is selected.

The model is a *portrait* segmenter — quality is best when people fill
its input. mask() therefore accepts an optional region of interest (the
pipeline passes a body-shaped expansion of the tracked faces): the net
sees just that region, and the result is pasted back into a full-frame
mask. Geometry follows the OpenCV-zoo reference (squashed resize in,
probability field resized up before any thresholding) and the
confidence gate is centred on 50% so the matte doesn't erode one-sided.

Temporal smoothing lives in the pipeline (full-frame space), so it stays
correct while the ROI follows a moving subject.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = _MODELS_DIR / "human_segmentation_pphumanseg_2023mar.onnx"
MODNET_PATH = _MODELS_DIR / "modnet_portrait.onnx"
RVM_PATH = _MODELS_DIR / "rvm_mobilenetv3_fp32.onnx"
_INPUT = 192


def _ort_session(path: Path):
    import onnxruntime as ort

    from .runtime import session_options
    if not path.exists():
        raise RuntimeError(f"{path.name} missing — run the launcher or "
                           "`python main.py --doctor` to download it")
    provs = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
             if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), sess_options=session_options(),
                                providers=provs)

# 25%..75% probability -> 0..255 alpha; the 50% midpoint lands on 128 so
# the cutout's re-harden threshold decides exactly like the zoo's argmax.
_GATE_LUT = np.clip((np.arange(256, dtype=np.float32) - 0.25 * 255) * 2.0,
                    0, 255).astype(np.uint8)


class PeopleSegmenter:
    soft = False  # coarse 192px mask: downstream re-hardens the edge

    def __init__(self, model_path: str | Path = MODEL_PATH):
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                "people-silhouette model missing — run the launcher or "
                "`python main.py --doctor` to download it")
        self.net = cv2.dnn.readNet(str(path))

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
        H, W = frame.shape[:2]
        if roi is None:
            return self._infer(frame)
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return self._infer(frame)
        out = np.zeros((H, W), dtype=np.uint8)
        out[y1:y2, x1:x2] = self._infer(frame[y1:y2, x1:x2])
        return out


class ModnetMatter:
    """MODNet portrait matting (Apache-2.0) via onnxruntime: true soft
    alpha with hair-level detail. Strongest on prominent single subjects
    (a presenter to camera); GPU-recommended, workable on CPU."""

    soft = True  # a real matte: don't re-harden the edge downstream

    def __init__(self, model_path: str | Path = MODNET_PATH, ref: int = 512):
        self.sess = _ort_session(Path(model_path))
        self.ref = ref

    def _infer(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        s = self.ref / max(H, W)
        nh = max(32, int(H * s) // 32 * 32)
        nw = max(32, int(W * s) // 32 * 32)
        x = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB) - 127.5) / 127.5
        m = self.sess.run(None, {"input": x.transpose(2, 0, 1)[None]})[0][0, 0]
        m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
        return cv2.resize(m8, (W, H), interpolation=cv2.INTER_LINEAR)

    def mask(self, frame: np.ndarray,
             roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        H, W = frame.shape[:2]
        if roi is None:
            return self._infer(frame)
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 32 or y2 - y1 < 32:
            return self._infer(frame)
        out = np.zeros((H, W), dtype=np.uint8)
        out[y1:y2, x1:x2] = self._infer(frame[y1:y2, x1:x2])
        return out


class RvmMatter:
    """Robust Video Matting (GPL-3.0) via onnxruntime: the best-looking
    people matte — temporally stable by design (recurrent state carries
    between frames). Runs full-frame; the ROI is ignored so the memory
    stays coherent. GPU-recommended; ~27ms/frame on an M-class CPU at
    720p, so workable for testing."""

    soft = True

    def __init__(self, model_path: str | Path = RVM_PATH):
        self.sess = _ort_session(Path(model_path))
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
# internal builds only (see facetrack/edition.py and LICENSE).
PEOPLE_MODELS = {
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

DEFAULT_PEOPLE_MODEL = "pphumanseg"


def _usable(key: str) -> bool:
    """A model is offered only if it may ship in this build AND is present."""
    from .edition import DISTRIBUTION
    info = PEOPLE_MODELS.get(key)
    if info is None or (DISTRIBUTION and not info["distributable"]):
        return False
    return Path(info["path"]).exists()


def available_people_models() -> list[dict]:
    """[{value, label}] for the panel — what this build can actually run."""
    return [{"value": k, "label": PEOPLE_MODELS[k]["label"]}
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
