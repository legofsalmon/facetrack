"""People segmentation for the silhouette cutout mode.

PP-HumanSeg (OpenCV model zoo, Apache-2.0): one combined matte of every
person in frame, 192x192 input via cv2.dnn — CPU-friendly, so it works
on both the Mac and the show machine. Loaded lazily the first time the
'people' cutout shape is selected.

Two quality measures beyond raw inference:
- the frame is letterboxed into the square net input (not squashed), so
  the mask maps back onto the picture without distortion or offset;
- the probability field is temporally smoothed (EMA) and median-filtered
  before gating, so static edges hold still instead of boiling.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "human_segmentation_pphumanseg_2023mar.onnx"
_INPUT = 192


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
        self._geom: tuple[int, int] | None = None

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame uint8 alpha (0..255) of everyone in the picture."""
        H, W = frame.shape[:2]
        # letterbox: fit the frame into the square input, keeping aspect
        s = _INPUT / max(W, H)
        iw, ih = max(1, round(W * s)), max(1, round(H * s))
        px, py = (_INPUT - iw) // 2, (_INPUT - ih) // 2
        canvas = np.zeros((_INPUT, _INPUT, 3), dtype=np.uint8)
        canvas[py:py + ih, px:px + iw] = cv2.resize(
            frame, (iw, ih), interpolation=cv2.INTER_AREA)

        blob = cv2.dnn.blobFromImage(canvas, 1 / 127.5, (_INPUT, _INPUT),
                                     (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()  # 1 x 2 x 192 x 192 logits (bg, person)
        logits = out[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        person = (e[1] / e.sum(axis=0)).astype(np.float32)
        person = person[py:py + ih, px:px + iw]  # drop the letterbox bars

        # temporal EMA: static edges stop flickering; motion lags a touch
        if self._geom != (W, H):
            self._geom = (W, H)
            self._ema = None
        if self.smoothing > 0 and self._ema is not None:
            person = self.smoothing * self._ema + (1 - self.smoothing) * person
        self._ema = person

        p8 = cv2.medianBlur((person * 255).astype(np.uint8), 3)
        # Gate the probability curve: below 35% is background noise, above
        # 85% is definitely person — mapping 35..85% to 0..255 kills the
        # milky low-confidence haze while keeping soft edges.
        alpha = np.clip((p8.astype(np.float32) - 0.35 * 255) * (255.0 / (0.5 * 255)),
                        0, 255).astype(np.uint8)
        return cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
