"""Lightweight SORT-style multi-face tracker.

Constant-velocity prediction + greedy IoU association, with a
center-distance fallback pass for small/fast-moving faces (IoU is brittle
when boxes are only a few pixels). No external dependencies; costs well
under a millisecond for hundreds of faces.
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of [x, y, w, h] boxes. Returns (len(a), len(b))."""
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    ix = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    iy = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = ix * iy
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = b[:, 2] * b[:, 3]
    return inter / (area_a + area_b - inter + 1e-9)


class Track:
    __slots__ = ("id", "bbox", "vel", "score", "hits", "misses", "age",
                 "emotion", "emotion_frame")

    def __init__(self, tid: int, bbox: np.ndarray, score: float):
        self.id = tid
        self.bbox = bbox.astype(np.float32).copy()  # x, y, w, h
        self.vel = np.zeros(2, dtype=np.float32)
        self.score = float(score)
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.emotion: tuple[str, float] | None = None
        self.emotion_frame = -(10 ** 9)

    @property
    def center(self) -> np.ndarray:
        return self.bbox[:2] + self.bbox[2:] * 0.5


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.25, max_misses: int = 15,
                 min_hits: int = 2, smooth: float = 0.5):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.smooth = smooth  # weight of the new detection in the bbox EMA
        self.tracks: list[Track] = []
        self._next_id = 1

    def step(self, detections: np.ndarray | None) -> list[Track]:
        """Advance one frame. detections is (N, 5) [x,y,w,h,score] or None
        when detection was skipped this frame (tracks coast on velocity)."""
        for t in self.tracks:
            t.age += 1
            t.bbox[:2] += t.vel
            t.vel *= 0.9

        if detections is None:
            return self.confirmed()

        dets = detections[:, :4]
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        if len(self.tracks) and len(dets):
            pred = np.stack([t.bbox for t in self.tracks])
            iou = iou_matrix(pred, dets)
            # Greedy pass on IoU.
            pairs = np.argwhere(iou >= self.iou_threshold)
            order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]]) if len(pairs) else []
            for k in order:
                ti, di = int(pairs[k, 0]), int(pairs[k, 1])
                if ti in matched_t or di in matched_d:
                    continue
                self._update(self.tracks[ti], dets[di], detections[di, 4])
                matched_t.add(ti)
                matched_d.add(di)
            # Fallback pass: center distance, for small faces where IoU fails.
            for ti, t in enumerate(self.tracks):
                if ti in matched_t:
                    continue
                gate = 0.75 * max(t.bbox[2], t.bbox[3])
                best_di, best_dist = -1, gate
                for di in range(len(dets)):
                    if di in matched_d:
                        continue
                    ratio = dets[di, 2] / (t.bbox[2] + 1e-9)
                    if not (0.5 <= ratio <= 2.0):
                        continue
                    dist = float(np.linalg.norm(t.center - (dets[di, :2] + dets[di, 2:] * 0.5)))
                    if dist < best_dist:
                        best_di, best_dist = di, dist
                if best_di >= 0:
                    self._update(t, dets[best_di], detections[best_di, 4])
                    matched_t.add(ti)
                    matched_d.add(best_di)

        for ti in range(len(self.tracks)):
            if ti not in matched_t:
                self.tracks[ti].misses += 1

        for di in range(len(dets)):
            if di not in matched_d:
                self.tracks.append(Track(self._next_id, dets[di], detections[di, 4]))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.confirmed()

    def _update(self, t: Track, det_bbox: np.ndarray, score: float) -> None:
        new_center = det_bbox[:2] + det_bbox[2:] * 0.5
        t.vel = 0.5 * t.vel + 0.5 * (new_center - t.center)
        a = self.smooth
        t.bbox = (1.0 - a) * t.bbox + a * det_bbox
        t.score = float(score)
        t.hits += 1
        t.misses = 0

    def confirmed(self) -> list[Track]:
        # misses gate uses the FULL max_misses window so the panel's
        # "Hold lost faces" seconds readout matches what's on screen
        return [t for t in self.tracks
                if t.hits >= self.min_hits and t.misses <= self.max_misses]
