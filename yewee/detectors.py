"""Face detection backends.

Two interchangeable backends, both returning an (N, 5) float32 array of
[x, y, w, h, score] in full-frame pixel coordinates:

- YuNetDetector: OpenCV's built-in YuNet. Very fast on CPU, the default on
  machines without an NVIDIA GPU (e.g. Apple Silicon).
- CenterFaceDetector: CenterFace (MIT) via ONNX Runtime. Fully
  convolutional, so unlike YuNet's fixed-640 export it scales to large
  input sizes — cheap on a CUDA/TensorRT provider, where it resolves
  smaller/more distant faces than YuNet can.

pick_backend() auto-selects CenterFace when a CUDA/TensorRT ONNX Runtime
provider is available, otherwise YuNet.

Both models are permissively licensed (MIT), so they can ship in a
distributed build — see LICENSE.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
CENTERFACE_MODEL = os.path.join(MODELS_DIR, "centerface_dynamic.onnx")

EMPTY = np.zeros((0, 5), dtype=np.float32)


class YuNetDetector:
    def __init__(self, model_path: str = YUNET_MODEL, input_width: int = 640,
                 score_threshold: float = 0.5, nms_threshold: float = 0.35, top_k: int = 1500):
        self.name = "yunet-cpu"
        self.model_path = model_path
        self.input_width = int(input_width)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self._det = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold, nms_threshold, top_k)
        self._size = None

    def apply_live(self, threshold: float, size: int) -> None:
        """Apply runtime-adjustable settings (safe to call every frame)."""
        self.input_width = max(160, int(size))
        if abs(float(threshold) - self.score_threshold) > 1e-6:
            self.score_threshold = float(threshold)
            try:
                self._det.setScoreThreshold(self.score_threshold)
            except AttributeError:
                self._det = cv2.FaceDetectorYN.create(
                    self.model_path, "", self._size or (320, 320),
                    self.score_threshold, self.nms_threshold, self.top_k)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.input_width / float(w))
        if scale < 1.0:
            dw, dh = int(round(w * scale)), int(round(h * scale))
            small = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            small, dw, dh, scale = frame_bgr, w, h, 1.0
        if self._size != (dw, dh):
            self._det.setInputSize((dw, dh))
            self._size = (dw, dh)
        _, faces = self._det.detect(small)
        if faces is None or len(faces) == 0:
            return EMPTY
        out = np.empty((len(faces), 5), dtype=np.float32)
        out[:, :4] = faces[:, :4] / scale
        out[:, 4] = faces[:, 14]
        return out


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= threshold]
    return keep


class CenterFaceDetector:
    """CenterFace via ONNX Runtime (CUDA/TensorRT on NVIDIA, CPU elsewhere).

    Anchor-free: the network emits a face-centre heatmap plus per-cell
    size and sub-pixel offset at stride 4, which we decode and NMS. The
    shipped model has dynamic input dims so `input_size` can be raised to
    resolve small faces — worthwhile on a GPU, slow on CPU.
    """

    def __init__(self, model_path: str = CENTERFACE_MODEL, input_size: int = 640,
                 score_threshold: float = 0.5, nms_threshold: float = 0.35,
                 providers: list[str] | None = None):
        from .runtime import make_session
        self.session = make_session(model_path, providers)
        active = self.session.get_providers()[0]
        self.name = "centerface-" + active.replace("ExecutionProvider", "").lower()
        self.input_name = self.session.get_inputs()[0].name
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = max(160, int(input_size))

    def apply_live(self, threshold: float, size: int) -> None:
        """Apply runtime-adjustable settings (safe to call every frame)."""
        self.score_threshold = float(threshold)
        self.input_size = max(160, int(size))

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        s = self.input_size / float(max(H, W))
        # the network needs both sides to be multiples of 32
        nh = max(32, int(np.ceil(H * s / 32) * 32))
        nw = max(32, int(np.ceil(W * s / 32) * 32))
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (nw, nh), (0, 0, 0),
                                     swapRB=True, crop=False)
        heat, scale, offset, _kps = self.session.run(None, {self.input_name: blob})
        hm = heat[0, 0]
        ys, xs = np.where(hm > self.score_threshold)
        if ys.size == 0:
            return EMPTY
        scale, offset = scale[0], offset[0]
        # decode: exp(scale) * stride gives the box, offset refines the centre
        bh = np.exp(scale[0, ys, xs]) * 4.0
        bw = np.exp(scale[1, ys, xs]) * 4.0
        cx = (xs + offset[1, ys, xs] + 0.5) * 4.0
        cy = (ys + offset[0, ys, xs] + 0.5) * 4.0
        rx, ry = W / nw, H / nh          # back to full-frame pixels
        x1 = (cx - bw / 2) * rx
        y1 = (cy - bh / 2) * ry
        bw, bh = bw * rx, bh * ry
        scores = hm[ys, xs].astype(np.float32)
        xyxy = np.stack([x1, y1, x1 + bw, y1 + bh], axis=1).astype(np.float32)
        keep = _nms(xyxy, scores, self.nms_threshold)
        out = np.empty((len(keep), 5), dtype=np.float32)
        out[:, 0] = np.clip(x1[keep], 0, W)
        out[:, 1] = np.clip(y1[keep], 0, H)
        out[:, 2] = bw[keep]
        out[:, 3] = bh[keep]
        out[:, 4] = scores[keep]
        return out


def pick_backend(backend: str, det_size: int, score_threshold: float):
    """backend: 'auto' | 'yunet' | 'centerface'."""
    if backend == "auto":
        try:
            import onnxruntime as ort
            listed = set(ort.get_available_providers())
        except ImportError:
            listed = set()
        if not ({"CUDAExecutionProvider", "TensorrtExecutionProvider"} & listed):
            return YuNetDetector(input_width=det_size, score_threshold=score_threshold)
        # Those providers are only listed, not proven — onnxruntime-gpu
        # advertises them on a machine with no CUDA installed. CenterFace
        # reports the provider it actually got, and on CPU it is slower than
        # YuNet, so a fallback to CPU means "auto" should not have chosen it.
        detector = CenterFaceDetector(input_size=det_size,
                                      score_threshold=score_threshold)
        if not detector.name.endswith("cpu"):
            return detector
        logging.getLogger("yewee").info(
            "no working GPU provider — using the CPU detector (YuNet) instead")
        return YuNetDetector(input_width=det_size, score_threshold=score_threshold)
    if backend == "centerface":
        return CenterFaceDetector(input_size=det_size, score_threshold=score_threshold)
    return YuNetDetector(input_width=det_size, score_threshold=score_threshold)
