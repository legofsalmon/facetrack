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

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / \
    "human_segmentation_pphumanseg_2023mar.onnx"
_INPUT = 192

# 25%..75% probability -> 0..255 alpha; the 50% midpoint lands on 128 so
# the cutout's re-harden threshold decides exactly like the zoo's argmax.
_GATE_LUT = np.clip((np.arange(256, dtype=np.float32) - 0.25 * 255) * 2.0,
                    0, 255).astype(np.uint8)


class PeopleSegmenter:
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
