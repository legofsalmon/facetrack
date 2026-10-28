"""Lightweight SORT-style multi-face tracker.

Constant-velocity prediction + greedy IoU association, with a
center-distance fallback pass for small/fast-moving faces (IoU is brittle
when boxes are only a few pixels). No external dependencies; costs well
under a millisecond for hundreds of faces.

Two things were measured rather than assumed, and the results are worth
recording because they are not what you would guess:

- Replacing the greedy association with a globally optimal one (a proper
  assignment solver over the same costs) changes nothing. On a synthetic
  crossing crowd it gives an identical switch count at 12 and 40 people
  and a slightly worse one at 120. Greedy over a sorted IoU list is
  already taking the best pair first, and on this cost that is enough.
  The benchmark lives in tests/idbench.py if you want to re-check.

- What does cost numbers is a face the detector loses for longer than
  the miss window and then finds again — someone turning away, or a
  dropout in a dense crowd. Every one of those used to mint a fresh
  number. Retired tracks are now held briefly and reclaimed on
  geometry, which is deliberately conservative: a wrong number handed
  back is worse than a new one, so a reclaim needs a matching size, a
  plausible position, and no rival candidate close behind.
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
                 "emotion", "emotion_frame", "retired_at")

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
        self.retired_at = 0

    @property
    def center(self) -> np.ndarray:
        return self.bbox[:2] + self.bbox[2:] * 0.5


class FaceTracker:
    #: How long a retired track stays reclaimable, in frames. Three
    #: seconds at 30 fps: long enough to cover someone turning away and
    #: back, short enough that a genuinely new arrival in the same spot
    #: is unlikely to inherit the number.
    REID_FRAMES = 90
    #: How far from where it was last seen a retired track may be
    #: reclaimed, as a multiple of its own size.
    REID_RADIUS = 2.5
    #: A reclaim needs a clear winner: the runner-up must be at least
    #: this much further away, or no number is handed back.
    REID_MARGIN = 1.4

    def __init__(self, iou_threshold: float = 0.25, max_misses: int = 15,
                 min_hits: int = 2, smooth: float = 0.5, reid: bool = True):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.smooth = smooth  # weight of the new detection in the bbox EMA
        self.reid = reid
        self.tracks: list[Track] = []
        self.retired: list[Track] = []   # recently lost, still reclaimable
        self._next_id = 1
        self._frame = 0

    def step(self, detections: np.ndarray | None) -> list[Track]:
        """Advance one frame. detections is (N, 5) [x,y,w,h,score] or None
        when detection was skipped this frame (tracks coast on velocity)."""
        self._frame += 1
        if self.retired:
            cutoff = self._frame - self.REID_FRAMES
            self.retired = [t for t in self.retired if t.retired_at > cutoff]
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
            # Fallback pass: center distance, for small faces where IoU
            # fails. Vectorised — as a pair of Python loops this was the
            # tracker's whole cost at crowd scale (7.9 ms for 200 faces
            # against 200 detections, with every track unmatched, which
            # is exactly where auto-relief takes you: detect_every goes
            # up, tracks coast, coasting makes IoU miss).
            rest_t = [ti for ti in range(len(self.tracks)) if ti not in matched_t]
            rest_d = [di for di in range(len(dets)) if di not in matched_d]
            if rest_t and rest_d:
                ti_ix = np.array(rest_t)
                di_ix = np.array(rest_d)
                tb, db = pred[ti_ix], dets[di_ix]
                tc = tb[:, :2] + tb[:, 2:] * 0.5
                dc = db[:, :2] + db[:, 2:] * 0.5
                dist = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2)
                gate = 0.75 * np.maximum(tb[:, 2], tb[:, 3])[:, None]
                ratio = db[None, :, 2] / (tb[:, 2][:, None] + 1e-9)
                dist = np.where((dist < gate) & (ratio >= 0.5) & (ratio <= 2.0),
                                dist, np.inf)
                for flat in np.argsort(dist, axis=None):
                    a, b = divmod(int(flat), dist.shape[1])
                    if not np.isfinite(dist[a, b]):
                        break               # sorted, so the rest are worse
                    ti, di = int(ti_ix[a]), int(di_ix[b])
                    if ti in matched_t or di in matched_d:
                        continue
                    self._update(self.tracks[ti], dets[di], detections[di, 4])
                    matched_t.add(ti)
                    matched_d.add(di)

        for ti in range(len(self.tracks)):
            if ti not in matched_t:
                self.tracks[ti].misses += 1

        for di in range(len(dets)):
            if di not in matched_d:
                self.tracks.append(self._born(dets[di], detections[di, 4]))

        lost = [t for t in self.tracks if t.misses > self.max_misses]
        if lost and self.reid:
            for t in lost:
                t.retired_at = self._frame
            self.retired.extend(lost)
        if lost:
            self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.confirmed()

    def _born(self, det_bbox: np.ndarray, score: float) -> Track:
        """A track for an unmatched detection: the number of a face we
        recently lost in about this spot, or a fresh one."""
        claimed = self._reclaim(det_bbox) if self.reid else None
        if claimed is None:
            track = Track(self._next_id, det_bbox, score)
            self._next_id += 1
            return track
        track = Track(claimed.id, det_bbox, score)
        # It is the same person, so don't make them earn confirmation
        # again — the number would blink off and on otherwise.
        track.hits = max(self.min_hits, claimed.hits)
        track.emotion = claimed.emotion
        return track

    def _reclaim(self, det_bbox: np.ndarray) -> Track | None:
        """The retired track this detection most likely belongs to.

        Conservative by design: handing back the wrong number is worse
        than issuing a new one. A candidate must be about the same size,
        near where it was last seen, and clearly nearer than any rival."""
        if not self.retired:
            return None
        centre = det_bbox[:2] + det_bbox[2:4] * 0.5
        best = second = None
        best_d = second_d = np.inf
        for r in self.retired:
            ratio = float(det_bbox[2] / (r.bbox[2] + 1e-9))
            if not (0.75 <= ratio <= 1.33):
                continue
            dist = float(np.linalg.norm(r.center - centre))
            if dist > self.REID_RADIUS * max(r.bbox[2], r.bbox[3]):
                continue
            if dist < best_d:
                second, second_d = best, best_d
                best, best_d = r, dist
            elif dist < second_d:
                second, second_d = r, dist
        if best is None:
            return None
        if second is not None and second_d < best_d * self.REID_MARGIN:
            return None          # two plausible people; don't guess
        self.retired.remove(best)
        return best

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
