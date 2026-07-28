"""Expression estimation with FER+ (emotion-ferplus-8.onnx) via OpenCV DNN.

The model is tiny (64x64 grayscale in, 8 scores out), so it runs on CPU.
To keep the frame budget flat regardless of crowd size, only up to
`budget_per_frame` faces are scored per frame, round-robin by staleness;
each track caches its last result.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .detectors import MODELS_DIR
from .tracker import Track

FERPLUS_MODEL = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

EMOTIONS = ("neutral", "happy", "surprise", "sad", "anger",
            "disgust", "fear", "contempt")


class EmotionEstimator:
    def __init__(self, model_path: str = FERPLUS_MODEL, budget_per_frame: int = 4,
                 refresh_interval: int = 12, min_face_px: int = 28, margin: float = 0.15):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.budget = int(budget_per_frame)
        self.refresh_interval = int(refresh_interval)
        self.min_face_px = int(min_face_px)
        self.margin = float(margin)

    def update(self, frame_bgr: np.ndarray, tracks: list[Track], frame_idx: int) -> None:
        H, W = frame_bgr.shape[:2]
        cands = [t for t in tracks
                 if t.bbox[2] >= self.min_face_px and t.bbox[3] >= self.min_face_px
                 and frame_idx - t.emotion_frame >= self.refresh_interval]
        cands.sort(key=lambda t: t.emotion_frame)  # stalest first
        for t in cands[:self.budget]:
            x, y, w, h = t.bbox
            mx, my = w * self.margin, h * self.margin
            x1 = max(0, int(x - mx))
            y1 = max(0, int(y - my))
            x2 = min(W, int(x + w + mx))
            y2 = min(H, int(y + h + my))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            blob = face.reshape(1, 1, 64, 64).astype(np.float32)
            self.net.setInput(blob)
            scores = self.net.forward().ravel()
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
            k = int(probs.argmax())
            t.emotion = (EMOTIONS[k], float(probs[k]))
            t.emotion_frame = frame_idx
