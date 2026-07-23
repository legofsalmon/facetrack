"""Face detection backends.

Two interchangeable backends, both returning an (N, 5) float32 array of
[x, y, w, h, score] in full-frame pixel coordinates:

- YuNetDetector: OpenCV's built-in YuNet. Very fast on CPU, the default on
  machines without an NVIDIA GPU (e.g. Apple Silicon).
- SCRFDDetector: InsightFace SCRFD-10G via ONNX Runtime. More accurate on
  dense crowds / small faces; on an NVIDIA GPU (CUDA/TensorRT provider) it
  runs in a few milliseconds even at large input sizes.

pick_backend() auto-selects SCRFD when a CUDA/TensorRT ONNX Runtime provider
is available, otherwise YuNet.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SCRFD_MODEL = os.path.join(MODELS_DIR, "scrfd_10g.onnx")

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


def _distance2bbox(centers: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return np.stack([
        centers[:, 0] - distances[:, 0],
        centers[:, 1] - distances[:, 1],
        centers[:, 0] + distances[:, 2],
        centers[:, 1] + distances[:, 3],
    ], axis=-1)


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


class SCRFDDetector:
    """SCRFD via ONNX Runtime (CUDA/TensorRT on NVIDIA, CPU elsewhere)."""

    def __init__(self, model_path: str = SCRFD_MODEL, input_size: int = 640,
                 score_threshold: float = 0.45, nms_threshold: float = 0.4,
                 providers: list[str] | None = None):
        import onnxruntime as ort
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("TensorrtExecutionProvider",
                                     "CUDAExecutionProvider",
                                     "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(model_path, providers=providers)
        active = self.session.get_providers()[0]
        self.name = "scrfd-" + active.replace("ExecutionProvider", "").lower()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.fmc = 3
        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        size = max(160, int(input_size))
        self.input_size = size - (size % 32)
        self._center_cache: dict[tuple, np.ndarray] = {}

    def apply_live(self, threshold: float, size: int) -> None:
        """Apply runtime-adjustable settings (safe to call every frame)."""
        self.score_threshold = float(threshold)
        size = max(160, int(size))
        self.input_size = size - (size % 32)

    def _centers(self, stride: int) -> np.ndarray:
        key = (self.input_size, stride)
        centers = self._center_cache.get(key)
        if centers is None:
            n = self.input_size // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            centers = np.stack([xv, yv], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self._center_cache[key] = centers
        return centers

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]
        s = self.input_size
        scale = min(s / float(W), s / float(H))
        dw, dh = int(round(W * scale)), int(round(H * scale))
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        canvas[:dh, :dw] = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (s, s),
                                     (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        all_scores, all_boxes = [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + self.fmc].reshape(-1, 4) * stride
            pos = np.where(scores >= self.score_threshold)[0]
            if pos.size == 0:
                continue
            boxes = _distance2bbox(self._centers(stride)[pos], bbox_preds[pos])
            all_scores.append(scores[pos])
            all_boxes.append(boxes)

        if not all_boxes:
            return EMPTY
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep = _nms(boxes, scores, self.nms_threshold)
        boxes, scores = boxes[keep] / scale, scores[keep]
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, H)
        out = np.empty((len(boxes), 5), dtype=np.float32)
        out[:, 0] = boxes[:, 0]
        out[:, 1] = boxes[:, 1]
        out[:, 2] = boxes[:, 2] - boxes[:, 0]
        out[:, 3] = boxes[:, 3] - boxes[:, 1]
        out[:, 4] = scores
        return out


def pick_backend(backend: str, det_size: int, score_threshold: float):
    """backend: 'auto' | 'yunet' | 'scrfd'."""
    if backend == "auto":
        try:
            import onnxruntime as ort
            gpu = {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & set(ort.get_available_providers())
            backend = "scrfd" if gpu else "yunet"
        except ImportError:
            backend = "yunet"
    if backend == "scrfd":
        return SCRFDDetector(input_size=det_size, score_threshold=score_threshold)
    return YuNetDetector(input_width=det_size, score_threshold=max(score_threshold, 0.5))
