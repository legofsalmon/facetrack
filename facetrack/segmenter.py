"""People segmentation for the silhouette cutout mode.

PP-HumanSeg (OpenCV model zoo, Apache-2.0): one combined matte of every
person in frame, 192x192 input via cv2.dnn — CPU-friendly, so it works
on both the Mac and the show machine. Loaded lazily the first time the
'people' cutout shape is selected.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "human_segmentation_pphumanseg_2023mar.onnx"
_INPUT = 192


class PeopleSegmenter:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                "people-silhouette model missing — run the launcher or "
                "`python main.py --doctor` to download it")
        self.net = cv2.dnn.readNet(str(path))

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """Full-frame uint8 alpha (0..255) of everyone in the picture."""
        blob = cv2.dnn.blobFromImage(frame, 1 / 127.5, (_INPUT, _INPUT),
                                     (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()  # 1 x 2 x 192 x 192 logits (bg, person)
        logits = out[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        person = e[1] / e.sum(axis=0)
        # Gate the probability curve: below 35% is background noise, above
        # 85% is definitely person — mapping 35..85% to 0..255 kills the
        # milky low-confidence haze while keeping soft edges.
        alpha = np.clip((person - 0.35) * (255.0 / 0.5), 0.0, 255.0)
        small = alpha.astype(np.uint8)
        return cv2.resize(small, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
