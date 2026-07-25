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


@run("capture: format strings parse safely")
def _():
    from facetrack.capture import parse_cap_format
    assert parse_cap_format("1920x1080@50") == (1920, 1080, 50.0)
    assert parse_cap_format("1280x720@29.97") == (1280, 720, 29.97)
    assert parse_cap_format("auto") == (0, 0, 0.0)
    assert parse_cap_format("") == (0, 0, 0.0)
    assert parse_cap_format("garbage@x") == (0, 0, 0.0)


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
    # hold time must match the panel's promise: a lost box survives
    # exactly max_misses frames of empty detections, then goes
    trk.max_misses = 10
    empty = np.zeros((0, 5), dtype=np.float32)
    for _ in range(10):
        held = trk.step(empty)
    assert len(held) == 3, "boxes must hold for the full max_misses window"
    assert len(trk.step(empty)) == 0, "boxes must drop right after the window"


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


@run("overlay: brand colour overrides the palette")
def _():
    from facetrack.overlay import render_overlay_bgra
    from facetrack.tracker import Track
    tracks = []
    for i in range(3):
        t = Track.__new__(Track)
        t.id = i
        t.bbox = (50 + i * 150, 60, 80, 80)
        t.emotion = None
        tracks.append(t)
    bgra = render_overlay_bgra((360, 640), tracks, color=(0, 0, 255))  # pure red
    drawn = bgra[bgra[:, :, 3] > 200]
    assert len(drawn), "nothing drawn"
    assert (drawn[:, 2].astype(int) >= drawn[:, 0].astype(int)).all(), \
        "brand colour must replace the palette (found blue-dominant pixels)"
    assert (drawn[:, 2].astype(int) >= drawn[:, 1].astype(int)).all(), \
        "brand colour must replace the palette (found green-dominant pixels)"


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


@run("mask feed: white-on-black and white-on-alpha styles")
def _():
    from facetrack.overlay import cutout_alpha, render_mask
    from facetrack.tracker import Track
    t = Track.__new__(Track)
    t.id = 0
    t.bbox = (100, 80, 60, 70)
    alpha = cutout_alpha((360, 640), [t], margin=0.1, shape="oval", feather=15)
    white = render_mask(alpha, "white")
    assert white.shape == (360, 640, 3)
    assert (white[:, :, 0] == alpha).all(), "white style must be the alpha as BGR"
    assert white[0, 0].max() == 0, "background must be black"
    av = render_mask(alpha, "alpha")
    assert av.shape == (360, 640, 4)
    assert (av[:, :, 3] == alpha).all()
    assert (av[:, :, 0] == alpha).all(), "premultiplied white silhouette"


@run("settings: old output params migrate to the feed matrix")
def _():
    from facetrack import settings
    with tempfile.TemporaryDirectory() as td:
        old = settings.SETTINGS_PATH
        settings.SETTINGS_PATH = Path(td) / "settings.json"
        try:
            settings.SETTINGS_PATH.write_text(
                '{"params": {"ndi_main": false, "texture_share": true,'
                ' "texture_source": "faces"}}')
            p = settings.load()["params"]
            assert p["ndi_program"] is False, "ndi_main must migrate"
            assert p["tex_faces"] is True, "texture_share+source must migrate"
        finally:
            settings.SETTINGS_PATH = old


@run("silhouette margin: grows and shrinks the people mask")
def _():
    from facetrack.overlay import cutout_alpha, grow_alpha
    disc = np.zeros((360, 640), dtype=np.uint8)
    cv2.circle(disc, (320, 180), 100, 255, -1)
    base = (disc > 127).sum()

    grown = (grow_alpha(disc, 12) > 127).sum()
    shrunk = (grow_alpha(disc, -12) > 127).sum()
    assert grown > base > shrunk, f"grow/shrink must change area ({shrunk} < {base} < {grown})"
    assert grow_alpha(disc, 0) is disc, "zero must be a no-op"

    # radius moves by roughly the requested pixels (area of a disc)
    r_grown = (grown / np.pi) ** 0.5
    assert 108 < r_grown < 116, f"expected ~112px radius after +12, got {r_grown:.1f}"

    # reaches the people path with a soft matte, edge detail preserved
    soft = cv2.GaussianBlur(disc, (21, 21), 0)
    a_wide = cutout_alpha((360, 640), [], shape="people", people_mask=soft,
                          people_soft=True, grow=10)
    a_tight = cutout_alpha((360, 640), [], shape="people", people_mask=soft,
                           people_soft=True, grow=-10)
    assert (a_wide > 127).sum() > (a_tight > 127).sum()
    assert ((a_wide > 0) & (a_wide < 255)).any(), "soft edge must survive the grow"


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


@run("matting models: MODNet and RVM produce sane soft mattes")
def _():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("        (onnxruntime not installed — skipped)")
        return
    from facetrack.overlay import render_faces_cutout
    from facetrack.segmenter import ModnetMatter, RvmMatter
    frame = _first_frame()
    for cls in (ModnetMatter, RvmMatter):
        model = cls()
        assert model.soft is True
        m = model.mask(frame)
        assert m.shape == frame.shape[:2] and m.dtype == np.uint8
        cut = render_faces_cutout(frame, [], shape="people", people_mask=m,
                                  people_soft=True)
        a = cut[:, :, 3]
        assert (cut[:, :, :3].astype(int) <= a[..., None].astype(int) + 1).all(), \
            f"{cls.__name__} cutout must stay premultiplied"
    # RVM: recurrent state survives repeat frames and resets on size change
    r = RvmMatter()
    r.mask(frame)
    r.mask(frame)
    small = cv2.resize(frame, (320, 180))
    assert r.mask(small).shape == (180, 320)


@run("people segmenter: ROI keeps the matte inside the region")
def _():
    from facetrack.segmenter import PeopleSegmenter
    frame = _first_frame()
    seg = PeopleSegmenter()
    H, W = frame.shape[:2]
    roi = (W // 4, H // 4, 3 * W // 4, 3 * H // 4)
    m = seg.mask(frame, roi=roi)
    assert m.shape == frame.shape[:2]
    outside = m.copy()
    outside[roi[1]:roi[3], roi[0]:roi[2]] = 0
    assert outside.max() == 0, "mask must be empty outside the ROI"
    # degenerate ROI falls back to full frame without crashing
    m2 = seg.mask(frame, roi=(0, 0, 4, 4))
    assert m2.shape == frame.shape[:2]


@run("runtime: CPU limit caps OpenCV and ONNX threads")
def _():
    import cv2 as _cv
    from facetrack import runtime
    default = _cv.getNumThreads()
    try:
        n = runtime.limit_threads(True)
        assert 1 <= n <= max(1, runtime.cores() // 2)
        # OpenCV only honours an arbitrary count on TBB/OpenMP/pthreads
        # builds; macOS GCD builds ignore it. Either is acceptable — the
        # ONNX cap below is the one that governs the expensive models.
        assert runtime.cv_threads() in (n, runtime.cores())
        so = runtime.session_options()
        if so is not None:  # None when onnxruntime isn't installed (CI)
            assert so.intra_op_num_threads == n
            assert so.inter_op_num_threads == max(1, n // 2)
        runtime.limit_threads(False)
        assert runtime.budget() == 0, "unlimited = library default"
        assert runtime.session_options() is None
    finally:
        runtime.limit_threads(False)
        _cv.setNumThreads(default)


@run("auto relief: steps down under load, recovers with headroom")
def _():
    from main import DEFAULTS
    from facetrack.params import LiveParams
    from facetrack.pipeline import Pipeline

    pipe = Pipeline.__new__(Pipeline)          # logic only, no capture/models
    pipe._relief = 0
    pipe._over_since = pipe._under_since = None
    pipe.last_error = ""
    pipe._error_time = 0.0
    on = {"auto_relief": True, "detect_every": 1, "det_size": 1280}

    t = 100.0
    pipe._update_relief(on, 150, t)            # first overload sample
    assert pipe._relief == 0, "must not react to a single spike"
    for step in (1, 2, 3):
        t += 6.0
        pipe._update_relief(on, 150, t)
        assert pipe._relief == step, f"expected step {step}, got {pipe._relief}"
    t += 6.0
    pipe._update_relief(on, 150, t)
    assert pipe._relief == 3, "must not exceed the last step"

    eff = pipe._relieved(on)
    assert eff["detect_every"] == 2 and eff["det_size"] == 640
    assert on["detect_every"] == 1, "operator's own settings must not be rewritten"

    t += 1.0
    pipe._update_relief(on, 40, t)             # first low sample starts the clock
    assert pipe._relief == 3, "must not restore on a single quiet sample"
    for expect in (2, 1, 0):                   # headroom sustained
        t += 21.0
        pipe._update_relief(on, 40, t)
        assert pipe._relief == expect, f"expected recovery to {expect}"

    pipe._relief = 2                           # switching it off clears state
    pipe._update_relief({**on, "auto_relief": False}, 200, t + 100)
    assert pipe._relief == 0


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
        if typ is bool:
            vals[k] = False
        elif typ is str:
            vals[k] = lo[0] if lo else ""  # free-form strings have no choices
        else:
            vals[k] = 1
    p = LiveParams(**vals)
    assert p.set("mask_style", "alpha") == "alpha"
    assert p.set("mask_style", "garbage") == "white"  # falls back
    assert p.set("overlay_color", "#ff8800") == "#ff8800"   # free string passes
    assert p.set("overlay_color", "") == ""


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
    params = LiveParams(**{**DEFAULTS, "ndi_program": False, "panel_preview": False,
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
