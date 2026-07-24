"""People segmentation for the silhouette cutout mode.

PP-HumanSeg (OpenCV model zoo, Apache-2.0): one combined matte of every
person in frame, 192x192 input via cv2.dnn — CPU-friendly, so it works
on both the Mac and the show machine. Loaded lazily the first time the
'people' cutout shape is selected.

Geometry follows the OpenCV-zoo reference exactly: plain (squashed)
resize in — the model was trained that way — and the probability field
is resized to full frame BEFORE any thresholding. The confidence gate is
centred on 50% (25..75% ramp): this model's confidence is asymmetric
around people, so an off-centre cut erodes one side of the matte and
makes it sit visibly offset from the subject.

The probability field is also temporally smoothed (EMA) and
median-filtered so static edges hold still instead of boiling.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "human_segmentation_pphumanseg_2023mar.onnx"
_INPUT = 192

# 25%..75% probability -> 0..255 alpha; the 50% midpoint lands on 128 so
# the cutout's re-harden threshold decides exactly like the zoo's argmax.
_GATE_LUT = np.clip((np.arange(256, dtype=np.float32) - 0.25 * 255) * 2.0,
                    0, 255).astype(np.uint8)


class PeopleSegmenter:
    def __init__(self, model_path: str | Path = MODEL_PATH,
                 smoothing: float = 0.55):
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                "people-silhouette model missing — run the launcher or "
                "`python main.py --doctor` to download it")
        self.net = cv2.dnn.readNet(str(path))
        self.smoothing = smoothing  # EMA weight on history; 0 disables
        self._ema: np.ndarray | None = None

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame uint8 alpha (0..255) of everyone in the picture."""
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (_INPUT, _INPUT), interpolation=cv2.INTER_AREA),
            1 / 127.5, (_INPUT, _INPUT), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()  # 1 x 2 x 192 x 192 logits (bg, person)
        logits = out[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        person = (e[1] / e.sum(axis=0)).astype(np.float32)

        # temporal EMA: static edges stop flickering; motion lags a touch
        if self.smoothing > 0 and self._ema is not None:
            person = self.smoothing * self._ema + (1 - self.smoothing) * person
        self._ema = person

        p8 = cv2.medianBlur((person * 255).astype(np.uint8), 3)
        full = cv2.resize(p8, (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.LUT(full, _GATE_LUT)
