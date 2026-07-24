"""Smoke tests — no test framework needed:

    .venv/bin/python -m tests.smoke

Covers params validation, settings persistence, the tracker, the YuNet
detector + overlay rendering on the committed test clip, and FER+
expression estimation. Avoids NDI, cameras, and the web server so it runs
anywhere (CI included).
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILURES: list[str] = []


def run(name):
    def wrap(fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            FAILURES.append(name)
    return wrap


@run("params: clamping and unknown keys")
def _():
    from facetrack.params import LiveParams, SPEC
    p = LiveParams(**{k: (False if SPEC[k][0] is bool else 1) for k in SPEC})
    assert p.set("det_threshold", 5.0) == 0.95
    assert p.set("det_threshold", -1) == 0.05
    assert p.set("emotion_enabled", 1) is True
    try:
        p.set("nope", 1)
        raise AssertionError("unknown key accepted")
    except KeyError:
        pass


@run("settings: roundtrip, pin and unknown keys preserved")
def _():
    from facetrack import settings
    with tempfile.TemporaryDirectory() as td:
        old = settings.SETTINGS_PATH
        settings.SETTINGS_PATH = Path(td) / "settings.json"
        try:
            settings.SETTINGS_PATH.write_text('{"pin": "4721", "custom": true}')
            settings.save(params={"det_threshold": 0.6}, source="1")
            data = settings.load()
            assert data["params"]["det_threshold"] == 0.6
            assert data["source"] == "1"
            assert data["pin"] == "4721"
            import json
            raw = json.loads(settings.SETTINGS_PATH.read_text())
            assert raw["custom"] is True, "unknown keys must survive writes"
        finally:
            settings.SETTINGS_PATH = old


@run("tracker: stable IDs on moving boxes")
def _():
    from facetrack.tracker import FaceTracker
    trk = FaceTracker()
    ids = set()
    for i in range(60):
        dets = np.array([
            [10 + i * 3, 20 + i * 2, 40, 40, 0.9],
            [300 - i * 2, 200, 36, 36, 0.9],
            [500, 50 + i * 3, 50, 50, 0.9],
        ], dtype=np.float32)
        tracks = trk.step(dets)
        ids.update(t.id for t in tracks)
    assert len(tracks) == 3, f"expected 3 confirmed tracks, got {len(tracks)}"
    assert len(ids) == 3, f"IDs churned: {sorted(ids)}"


def _first_frame():
    cap = cv2.VideoCapture(os.path.join(ROOT, "test_media", "synth.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok, "could not read test clip"
    return frame


@run("detector: YuNet finds the synthetic faces")
def _():
    from facetrack.detectors import YuNetDetector
    dets = YuNetDetector(score_threshold=0.4).detect(_first_frame())
    assert len(dets) >= 6, f"expected >=6 faces, got {len(dets)}"
    assert (dets[:, 4] >= 0.4).all()


@run("overlay: alpha only where graphics are drawn")
def _():
    from facetrack.detectors import YuNetDetector
    from facetrack.overlay import render_overlay_bgra
    from facetrack.tracker import FaceTracker
    frame = _first_frame()
    trk = FaceTracker(min_hits=1)
    tracks = trk.step(YuNetDetector(score_threshold=0.4).detect(frame))
    bgra = render_overlay_bgra(frame.shape[:2], tracks)
    alpha = bgra[:, :, 3]
    frac = (alpha > 0).mean()
    assert 0.001 < frac < 0.4, f"odd alpha coverage {frac:.3f}"
    assert bgra[alpha == 0][:, :3].max() == 0, "transparent pixels must be black"


@run("faces cutout: picture inside boxes, transparent outside")
def _():
    from facetrack.detectors import YuNetDetector
    from facetrack.overlay import render_faces_cutout
    from facetrack.tracker import FaceTracker
    frame = _first_frame()
    trk = FaceTracker(min_hits=1)
    tracks = trk.step(YuNetDetector(score_threshold=0.4).detect(frame))
    assert tracks, "need tracks for the cutout test"
    cut = render_faces_cutout(frame, tracks, margin=0.0)
    alpha = cut[:, :, 3]
    frac = (alpha > 0).mean()
    assert 0.005 < frac < 0.6, f"odd cutout coverage {frac:.3f}"
    for t in tracks:
        x, y, w, h = t.bbox
        cy, cx = int(y + h / 2), int(x + w / 2)
        assert alpha[cy, cx] == 255
        assert (cut[cy, cx, :3] == frame[cy, cx]).all(), "pixels must pass through"
    assert cut[alpha == 0].max() == 0, "transparent area must be empty"
    # margin grows the boxes
    grown = (render_faces_cutout(frame, tracks, margin=0.3)[:, :, 3] > 0).mean()
    assert grown > frac


@run("cutout shapes: ovals, feathering, premultiplied alpha")
def _():
    from facetrack.detectors import YuNetDetector
    from facetrack.overlay import render_faces_cutout
    from facetrack.tracker import FaceTracker
    frame = _first_frame()
    trk = FaceTracker(min_hits=1)
    tracks = trk.step(YuNetDetector(score_threshold=0.4).detect(frame))
    assert tracks

    oval = render_faces_cutout(frame, tracks, margin=0.1, shape="oval")
    a = oval[:, :, 3]
    t = tracks[0]
    x, y, w, h = t.bbox
    cy, cx = int(y + h / 2), int(x + w / 2)
    assert a[cy, cx] == 255, "oval centre must be opaque"
    # an oval leaves the box corners transparent (rectangle would not)
    rect = render_faces_cutout(frame, tracks, margin=0.1, shape="rectangle")
    assert (a > 0).sum() < (rect[:, :, 3] > 0).sum(), "oval must cover less than rect"

    soft = render_faces_cutout(frame, tracks, margin=0.1, shape="oval", feather=21)
    sa = soft[:, :, 3]
    assert ((sa > 0) & (sa < 255)).any(), "feather must create soft edges"
    assert (soft[:, :, :3].astype(int) <= sa[..., None].astype(int) + 1).all(), \
        "premultiplied: no channel may exceed alpha"


@run("people cutout: feather slider actually controls edge width")
def _():
    from facetrack.overlay import render_faces_cutout
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    disc = np.zeros((360, 640), dtype=np.uint8)
    cv2.circle(disc, (320, 180), 100, 255, -1)

    def soft_pixels(feather):
        a = render_faces_cutout(frame, [], shape="people", feather=feather,
                                people_mask=disc)[:, :, 3]
        return ((a > 32) & (a < 224)).sum()

    crisp, soft = soft_pixels(0), soft_pixels(40)
    assert soft > crisp * 4, f"feather 40 must widen the edge (crisp {crisp}, soft {soft})"
    # a mushy pre-blurred mask must still come out crisp at feather 0
    mush = cv2.GaussianBlur(disc, (31, 31), 0)
    a0 = render_faces_cutout(frame, [], shape="people", feather=0,
                             people_mask=mush)[:, :, 3]
    band = ((a0 > 32) & (a0 < 224)).sum()
    assert band < crisp * 3, f"feather 0 must re-harden a mushy mask (band {band})"


@run("people segmenter: loads and produces a full-frame mask")
def _():
    from facetrack.overlay import render_faces_cutout
    from facetrack.segmenter import PeopleSegmenter
    frame = _first_frame()
    seg = PeopleSegmenter()
    mask = seg.mask(frame)
    assert mask.shape == frame.shape[:2] and mask.dtype == np.uint8
    cut = render_faces_cutout(frame, [], shape="people", feather=10,
                              people_mask=mask)
    ca = cut[:, :, 3]
    assert (cut[:, :, :3].astype(int) <= ca[..., None].astype(int) + 1).all(), \
        "people cutout must be premultiplied"


@run("people segmenter: temporal smoothing steadies noisy edges")
def _():
    from facetrack.segmenter import PeopleSegmenter
    frame = _first_frame()
    rng = np.random.default_rng(7)

    def jitter(seg, n=6):
        """mean frame-to-frame mask change under simulated sensor noise"""
        prev, diffs = None, []
        for _ in range(n):
            noisy = cv2.add(frame, rng.integers(-12, 13, frame.shape,
                                                dtype=np.int16).astype(np.int16),
                            dtype=cv2.CV_8U)
            m = seg.mask(noisy).astype(np.int16)
            if prev is not None:
                diffs.append(np.abs(m - prev).mean())
            prev = m
        return float(np.mean(diffs))

    raw = jitter(PeopleSegmenter(smoothing=0.0))
    smooth = jitter(PeopleSegmenter(smoothing=0.55))
    assert smooth < raw * 0.75, f"EMA must reduce jitter (raw {raw:.3f}, smooth {smooth:.3f})"


@run("test card: bars on program, markers on alpha")
def _():
    from facetrack.overlay import render_test_card
    card, ovl = render_test_card(640, 360, ["facetrack TEST CARD", "MAC", "640x360"])
    assert card.shape == (360, 640, 3) and ovl.shape == (360, 640, 4)
    bar_row = card[10]
    uniques = len(np.unique(bar_row.reshape(-1, 3), axis=0))
    assert uniques >= 7, f"expected 7 colour bars, saw {uniques} colours"
    ramp_row = card[int(360 * 0.65)]
    assert ramp_row[..., 0].max() - ramp_row[..., 0].min() > 200, "ramp missing"
    alpha = ovl[:, :, 3]
    assert 0.001 < (alpha > 0).mean() < 0.2, "alpha test pattern coverage odd"
    assert ovl[alpha == 0].max() == 0, "alpha card must be empty where transparent"


@run("params: string choices validate")
def _():
    from facetrack.params import LiveParams, SPEC
    vals = {}
    for k, (typ, lo, _hi) in SPEC.items():
        vals[k] = False if typ is bool else (lo[0] if typ is str else 1)
    p = LiveParams(**vals)
    assert p.set("texture_source", "faces") == "faces"
    assert p.set("texture_source", "garbage") == "program"  # falls back


@run("pipeline: NO SIGNAL slate and recovery on dead live source")
def _():
    import threading
    import time

    from main import DEFAULTS, parse_args
    from facetrack.params import LiveParams
    from facetrack.pipeline import Pipeline

    class FlakySource:
        is_live = True
        fps = 30.0

        def __init__(self):
            self.dead = False
            self._frame = np.zeros((120, 160, 3), dtype=np.uint8)

        def read(self, timeout: float = 0.0):
            time.sleep(0.01)
            if self.dead:
                time.sleep(min(timeout, 0.05))
                return False, None
            return True, self._frame.copy()

        def close(self):
            pass

    args = parse_args(["--source", os.path.join(ROOT, "test_media", "synth.mp4"),
                       "--no-ndi", "--no-preview", "--no-web", "--no-browser",
                       "--quiet", "--backend", "yunet"])
    params = LiveParams(**{**DEFAULTS, "ndi_main": False, "panel_preview": False,
                           "local_preview": False, "emotion_enabled": False})
    pipe = Pipeline(args, params, web_enabled=False)
    pipe.source.close()
    flaky = FlakySource()
    pipe.source = flaky
    pipe.source_spec = "/nonexistent/dead-input"  # reopen attempts must fail
    t = threading.Thread(target=pipe.run, daemon=True)
    t.start()
    try:
        def wait_state(want, timeout):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if pipe.get_stats().get("state") == want:
                    return True
                time.sleep(0.05)
            return False

        assert wait_state("live", 5), "pipeline never went live"
        flaky.dead = True
        assert wait_state("no-signal", 8), "signal loss not detected"
        assert "Signal lost" in pipe.get_stats().get("error", "")
        black, _ = pipe._standby_frames("")  # what outputs carry during loss
        assert black.max() == 0, "live outputs must show plain black on signal loss"
        flaky.dead = False
        assert wait_state("live", 5), "did not recover when source returned"
        assert "Signal lost" not in pipe.get_stats().get("error", "")
    finally:
        pipe.stop()
        t.join(timeout=5)


@run("emotion: FER+ labels a face")
def _():
    from facetrack.detectors import YuNetDetector
    from facetrack.emotion import EMOTIONS, EmotionEstimator
    from facetrack.tracker import FaceTracker
    frame = _first_frame()
    trk = FaceTracker(min_hits=1)
    tracks = trk.step(YuNetDetector(score_threshold=0.4).detect(frame))
    est = EmotionEstimator(budget_per_frame=2)
    est.update(frame, tracks, 100)
    labelled = [t for t in tracks if t.emotion is not None]
    assert labelled, "no track got an emotion label"
    assert all(t.emotion[0] in EMOTIONS for t in labelled)


if FAILURES:
    print(f"\n{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("\nAll smoke tests passed.")
