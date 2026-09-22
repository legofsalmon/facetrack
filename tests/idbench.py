"""Track-identity benchmark: a synthetic crowd with known identities.

Real crowd footage would be better and we have none, so this drives the
tracker directly with detections whose true owner is known and counts
how often a person's number changes. Driving the tracker rather than the
whole pipeline is deliberate: it isolates the association logic from
detector noise and runs in a second, so it can guard a release.

Three scene kinds, because they fail for different reasons:

  bounce   everyone stays in shot, walking at different speeds and
           heights so they cross constantly. An ID change here is an
           association error.
  dropout  the detector loses people for longer than the miss window
           and finds them again where they walked to. An ID change here
           is a re-identification miss — the common one at an event,
           where somebody turns away and turns back.
  return   people walk off one side and come back on the other. Nothing
           geometric can join those up; it is here to show what
           appearance-based re-identification would have to solve, and
           to catch a reclaim rule that got reckless enough to guess.

Run it directly to print the table:  python -m tests.idbench
"""
from __future__ import annotations

import statistics as st
import time

import numpy as np

from yewee.tracker import FaceTracker

W, H = 1920, 1080


def scene(n_people: int = 12, frames: int = 300, seed: int = 0,
          miss_rate: float = 0.10, jitter: float = 1.5, kind: str = "bounce",
          detect_every: int = 1):
    """Yields (detections, true owner per detection) frame by frame."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(60, 130, n_people)
    y0 = rng.uniform(150, H - 300, n_people)
    x = rng.uniform(100, W - 100, n_people)
    vx = rng.choice([-1.0, 1.0], n_people) * rng.uniform(3, 11, n_people)
    bob = rng.uniform(0.02, 0.06, n_people)
    for f in range(frames):
        x = x + vx
        if kind == "return":
            x = (x + 300) % (W + 600) - 300
        else:
            turn = (x < 60) | (x > W - 60)
            vx = np.where(turn, -vx, vx)
            x = np.clip(x, 60, W - 60)
        if detect_every > 1 and f % detect_every:
            yield None, []            # a coasting frame: tracks predict
            continue
        boxes, owners = [], []
        for i in range(n_people):
            if not (-50 < x[i] < W + 50):
                continue              # genuinely out of shot
            if kind == "dropout" and (f // 40 + i) % 5 == 0 and f % 40 < 25:
                continue              # lost for 25 frames, then found again
            if rng.random() < miss_rate:
                continue              # the detector simply missed one
            y = y0[i] + np.sin(f * bob[i]) * 30
            boxes.append([x[i] + rng.normal(0, jitter), y + rng.normal(0, jitter),
                          size[i], size[i] * 1.2, 0.9])
            owners.append(i)
        yield (np.array(boxes, np.float32) if boxes
               else np.zeros((0, 5), np.float32)), owners


def evaluate(make_tracker=FaceTracker, **kw) -> dict:
    """Run one scene. -> {switches, matched, mean_ms, max_ms}."""
    trk = make_tracker()
    owner_of: dict[int, int] = {}
    switches = matched = 0
    times: list[float] = []
    for dets, owners in scene(**kw):
        t0 = time.perf_counter()
        tracks = trk.step(dets)
        times.append((time.perf_counter() - t0) * 1000)
        if dets is None or not len(dets) or not tracks:
            continue
        centres = np.stack([t.center for t in tracks])
        for owner, det in zip(owners, dets):
            centre = det[:2] + det[2:4] * 0.5
            d = np.linalg.norm(centres - centre, axis=1)
            k = int(d.argmin())
            if d[k] > det[2]:
                continue              # no track plausibly covers this person
            matched += 1
            was = owner_of.get(owner)
            if was is not None and was != tracks[k].id:
                switches += 1
            owner_of[owner] = tracks[k].id
    return {"switches": switches, "matched": matched,
            "mean_ms": st.mean(times), "max_ms": max(times)}


SCENES = [
    ("12 crossing, 10% misses", dict(n_people=12)),
    ("12 crossing, 30% misses", dict(n_people=12, miss_rate=0.30)),
    ("12 crossing, coasting 3", dict(n_people=12, detect_every=3)),
    ("40 crossing, 10% misses", dict(n_people=40)),
    ("120 crossing, 10% misses", dict(n_people=120, frames=150)),
    ("12 with detector dropouts", dict(n_people=12, kind="dropout", frames=400)),
    ("40 with detector dropouts", dict(n_people=40, kind="dropout", frames=400)),
    ("12 leaving and returning", dict(n_people=12, kind="return", frames=400)),
]


def unmatched_cost(n: int, repeats: int = 20) -> float:
    """Mean ms for a frame where no track matches any detection by IoU,
    so the centre-distance fallback runs across the whole crowd. This is
    where auto-relief takes you: detect_every rises, tracks coast,
    coasting makes IoU miss."""
    from yewee.tracker import Track
    rng = np.random.default_rng(0)
    trk = FaceTracker(reid=False)
    dets = np.column_stack([
        rng.uniform(0, W, n), rng.uniform(0, H, n),
        np.full(n, 80.0), np.full(n, 96.0), np.full(n, 0.9)]).astype(np.float32)
    times = []
    for _ in range(repeats):
        trk.tracks = [Track(i + 1, np.array([rng.uniform(0, W), rng.uniform(0, H),
                                             80.0, 96.0], np.float32), 0.9)
                      for i in range(n)]
        for t in trk.tracks:
            t.hits = 5
        t0 = time.perf_counter()
        trk.step(dets)
        times.append((time.perf_counter() - t0) * 1000)
    return st.mean(times)


def main() -> None:
    print(f"{'scene':<28}{'switches':>9}{'matched':>9}{'mean ms':>9}{'max ms':>9}")
    for label, kw in SCENES:
        r = evaluate(**kw)
        print(f"{label:<28}{r['switches']:>9}{r['matched']:>9}"
              f"{r['mean_ms']:>9.2f}{r['max_ms']:>9.2f}")
    print(f"\n{'nothing matches by IoU':<28}{'':>9}{'':>9}{'mean ms':>9}")
    for n in (40, 120, 200):
        print(f"{f'  {n} tracks x {n} faces':<28}{'':>9}{'':>9}{unmatched_cost(n):>9.2f}")


if __name__ == "__main__":
    main()
