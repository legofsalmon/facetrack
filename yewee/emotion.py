"""Expression estimation with FER+ (emotion-ferplus-8.onnx) via OpenCV DNN.

The model is tiny (64x64 grayscale in, 8 scores out) but not free: one
forward pass measures about 7.6 ms on a CPU, so the default budget of
four faces was up to 30 ms a frame — most of a 30 fps frame spent on
labels that only refresh every twelve frames. Batching the faces into
one pass would be the obvious fix and the model refuses it: its MatMul
has a fixed batch dimension.

So scoring runs on its own thread and the show loop never waits for it.
The crops are taken on the calling thread, which costs microseconds and
means the worker never reads a frame the pipeline is drawing on. Only
one batch is ever in flight; while it is, no more are dispatched, so a
slow machine falls behind on label freshness rather than on frame rate.

To keep the cost bounded regardless of crowd size, only up to
`budget_per_frame` faces are dispatched at a time, round-robin by
staleness; each track caches its last result.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .detectors import MODELS_DIR
from .tracker import Track

FERPLUS_MODEL = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

EMOTIONS = ("neutral", "happy", "surprise", "sad", "anger",
            "disgust", "fear", "contempt")


class EmotionEstimator:
    def __init__(self, model_path: str = FERPLUS_MODEL, budget_per_frame: int = 4,
                 refresh_interval: int = 12, min_face_px: int = 28, margin: float = 0.15,
                 threaded: bool = True):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.budget = int(budget_per_frame)
        self.refresh_interval = int(refresh_interval)
        self.min_face_px = int(min_face_px)
        self.margin = float(margin)
        self.threaded = bool(threaded)

        self._cond = threading.Condition()
        self._jobs: list | None = None   # the one batch in flight, if any
        self._busy = False
        self._stop = False
        self._error: Exception | None = None
        self._worker: threading.Thread | None = None

    # ---- the show loop's side ----

    def update(self, frame_bgr: np.ndarray, tracks: list[Track], frame_idx: int) -> None:
        """Dispatch stale faces for scoring. Returns immediately.

        An exception from the worker is re-raised here, on the pipeline's
        thread, so the existing handling (disable expressions, say so in
        the panel) works exactly as it did when this ran inline."""
        with self._cond:
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            if self._busy:
                return  # a batch is still being scored; don't pile on

        jobs = self._crop(frame_bgr, tracks, frame_idx)
        if not jobs:
            return
        if not self.threaded:
            self._score(jobs)
            return
        with self._cond:
            self._jobs = jobs
            self._busy = True
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, daemon=True,
                                                name="yewee-expressions")
                self._worker.start()
            self._cond.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until nothing is in flight. For tests and shutdown."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._busy, timeout=timeout)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(timeout=2.0)

    # ---- internals ----

    def _crop(self, frame_bgr: np.ndarray, tracks: list[Track],
              frame_idx: int) -> list:
        """The blobs for the stalest faces worth scoring, cropped now so
        the worker never touches the pipeline's frame."""
        H, W = frame_bgr.shape[:2]
        cands = [t for t in tracks
                 if t.bbox[2] >= self.min_face_px and t.bbox[3] >= self.min_face_px
                 and frame_idx - t.emotion_frame >= self.refresh_interval]
        cands.sort(key=lambda t: t.emotion_frame)  # stalest first
        jobs = []
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
            jobs.append((t, face.reshape(1, 1, 64, 64).astype(np.float32)))
            # Claim the face at dispatch, not at completion, or the next
            # few frames would queue the same faces again.
            t.emotion_frame = frame_idx
        return jobs

    def _score(self, jobs: list) -> None:
        for t, blob in jobs:
            self.net.setInput(blob)
            scores = self.net.forward().ravel()
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
            k = int(probs.argmax())
            # A plain attribute write, read by the drawing code on the
            # other thread. Rebinding one reference needs no lock.
            t.emotion = (EMOTIONS[k], float(probs[k]))

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._jobs is not None or self._stop)
                if self._stop:
                    return
                jobs, self._jobs = self._jobs, None
            try:
                self._score(jobs)
            except Exception as exc:          # noqa: BLE001 — reported upstream
                with self._cond:
                    self._error = exc
            with self._cond:
                self._busy = False
                self._cond.notify_all()
