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
