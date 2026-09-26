"""Smoke tests — no test framework needed:

    .venv/bin/python -m tests.smoke

Covers params validation, settings persistence, the tracker, the YuNet
detector + overlay rendering on the committed test clip, and FER+
expression estimation. Avoids NDI, cameras, and the web server so it runs
anywhere (CI included).
"""
from __future__ import annotations

import json
import logging
import os
import re
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
    from yewee.params import LiveParams, SPEC
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
    from yewee import settings
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
    from yewee.capture import parse_cap_format
    assert parse_cap_format("1920x1080@50") == (1920, 1080, 50.0)
    assert parse_cap_format("1280x720@29.97") == (1280, 720, 29.97)
    assert parse_cap_format("auto") == (0, 0, 0.0)
    assert parse_cap_format("") == (0, 0, 0.0)
    assert parse_cap_format("garbage@x") == (0, 0, 0.0)


@run("tracker: stable IDs on moving boxes")
def _():
    from yewee.tracker import FaceTracker
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
    from yewee.detectors import YuNetDetector
    dets = YuNetDetector(score_threshold=0.4).detect(_first_frame())
    assert len(dets) >= 6, f"expected >=6 faces, got {len(dets)}"
    assert (dets[:, 4] >= 0.4).all()


@run("overlay: alpha only where graphics are drawn")
def _():
    from yewee.detectors import YuNetDetector
    from yewee.overlay import render_overlay_bgra
    from yewee.tracker import FaceTracker
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
    from yewee.overlay import render_overlay_bgra
    from yewee.tracker import Track
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
    from yewee.detectors import YuNetDetector
    from yewee.overlay import render_faces_cutout
    from yewee.tracker import FaceTracker
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
    from yewee.detectors import YuNetDetector
    from yewee.overlay import render_faces_cutout
    from yewee.tracker import FaceTracker
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
    from yewee.overlay import cutout_alpha, render_mask
    from yewee.tracker import Track
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
    from yewee import settings
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
    from yewee.overlay import cutout_alpha, grow_alpha
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
    from yewee.overlay import render_faces_cutout
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
    from yewee.overlay import render_faces_cutout
    from yewee.segmenter import PeopleSegmenter
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
    from yewee.overlay import render_faces_cutout
    from yewee.segmenter import ModnetMatter, RvmMatter
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
    from yewee.segmenter import PeopleSegmenter
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


@run("detector: panel choice switches engine, bad choice falls back")
def _():
    from main import DEFAULTS
    from yewee.params import LiveParams
    from yewee.pipeline import Pipeline

    params = LiveParams(**{**DEFAULTS, "detector": "auto"})
    pipe = Pipeline.__new__(Pipeline)      # detector logic only
    pipe.params = params
    pipe.last_error = ""
    pipe._error_time = 0.0
    pipe._detector_choice = "auto"
    pipe.detector = None

    p = params.snapshot()
    pipe._sync_detector({**p, "detector": "yunet"})
    assert pipe._detector_choice == "yunet"
    assert pipe.detector is not None and "yunet" in pipe.detector.name

    # a backend that cannot load must fall back, not raise
    params.set("detector", "scrfd")
    pipe._sync_detector({**p, "detector": "scrfd"})
    assert pipe._detector_choice in ("scrfd", "yunet")
    if pipe._detector_choice == "yunet":       # no GPU runtime here
        assert params.snapshot()["detector"] == "yunet", "param must follow the fallback"
        assert "unavailable" in pipe.last_error
    assert pipe.detector is not None, "must always end with a working detector"


@run("params: launch-only flags stay out of the panel")
def _():
    from yewee.params import SPEC
    # everything an operator can change at runtime should be a param
    for key in ("detector", "out_fps", "loop_file", "cap_format", "cap_backend"):
        assert key in SPEC, f"{key} should be panel-controllable"
    # ...and structural/security settings should NOT be
    for key in ("web_port", "web_host", "pin", "ndi_name"):
        assert key not in SPEC, f"{key} must stay launch-only"


@run("ed25519: matches the RFC 8032 test vectors")
def _():
    from yewee import _ed25519 as ed
    vectors = [
        ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
         "",
         "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
         "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
        ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
         "72",
         "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da08"
         "5ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
        ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
         "af82",
         "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18"
         "ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
    ]
    for sk, pk, msg, sig in vectors:
        sk, pk = bytes.fromhex(sk), bytes.fromhex(pk)
        msg, sig = bytes.fromhex(msg), bytes.fromhex(sig)
        assert ed.public_key(sk) == pk
        assert ed.sign(sk, msg) == sig
        assert ed.verify(pk, msg, sig)
        assert not ed.verify(pk, msg + b"!", sig), "tampered message accepted"
        bad = bytearray(sig); bad[0] ^= 1
        assert not ed.verify(pk, msg, bytes(bad)), "tampered signature accepted"


@run("licensing: keys verify, expire, bind, and reject tampering")
def _():
    import secrets as _s
    from yewee import _ed25519 as ed, licensing as lic
    secret = _s.token_bytes(32)
    pub = ed.public_key(secret).hex()
    other = ed.public_key(_s.token_bytes(32)).hex()
    from datetime import date, timedelta

    def key(**over):
        payload = {"v": 1, "p": "yewee", "e": "pro", "n": "Test",
                   "i": date.today().isoformat(), "k": "abc123"}
        payload.update(over)
        return lic.encode_key(payload, secret)

    good = key()
    assert lic.decode_key(good, public_key_hex=pub)["n"] == "Test"
    assert lic.decode_key(good, public_key_hex=other) is None, "wrong key accepted"
    assert lic.decode_key("YW1.nonsense.nonsense", public_key_hex=pub) is None
    assert lic.decode_key("", public_key_hex=pub) is None
    # a flipped payload byte must fail the signature
    head, body, sig = good.split(".", 2)
    tampered = f"{head}.{body[:-2]}AA.{sig}"
    assert lic.decode_key(tampered, public_key_hex=pub) is None, "tampered payload accepted"
    # wrong product is refused even when correctly signed
    assert lic.decode_key(lic.encode_key({"p": "other"}, secret),
                          public_key_hex=pub) is None
    # optional fields survive the round trip
    dated = lic.decode_key(key(x=(date.today() + timedelta(days=30)).isoformat(),
                               m="deadbeef"), public_key_hex=pub)
    assert dated["m"] == "deadbeef" and dated["x"]


@run("paths: a packaged app never writes inside its own bundle")
def _():
    import sys
    from yewee import paths
    root = Path(paths._source_root())

    assert not paths.is_frozen()
    assert Path(paths.settings_path()).parent == root, "source runs stay in the checkout"
    assert Path(paths.log_dir()).parent == root

    frozen = getattr(sys, "frozen", None)
    try:                                   # pretend to be a PyInstaller build
        sys.frozen = True
        assert paths.is_frozen()
        for p in (Path(paths.settings_path()), Path(paths.log_path())):
            assert root not in p.parents and p != root, \
                f"{p} would be written inside the bundle"
            assert Path(paths.user_data_dir()) in p.parents
    finally:
        if frozen is None:
            del sys.frozen
        else:
            sys.frozen = frozen


@run("camera: permission is settled before anything opens a camera")
def _():
    import os
    import sys
    from yewee import capture

    # OpenCV would otherwise try to ask for permission from the capture
    # thread, where it cannot work.
    assert os.environ.get("OPENCV_AVFOUNDATION_SKIP_AUTH") == "1"

    # The request must report the outcome rather than fire and forget —
    # returning before the user answers means opening the camera too early.
    logs = []
    handler = logging.Handler()
    handler.emit = lambda r: logs.append(r.getMessage())
    log = logging.getLogger("yewee")
    log.addHandler(handler)
    try:
        assert capture.request_camera_access(timeout=1.0) in (
            "authorized", "denied", "undetermined", "restricted", "unknown")
    finally:
        log.removeHandler(handler)

    # It asked macOS for real. PyObjC refuses a Python callable where a block
    # is expected unless the selector's signature is registered, and for a
    # long time that failure was swallowed — so the prompt never appeared and
    # the camera could never be authorised. Never again silently.
    assert not any("camera permission" in m for m in logs), \
        f"the permission request failed: {logs}"

    if sys.platform == "darwin":
        holder = capture.camera_permission_holder()
        assert holder == "your terminal app", "source runs hold no permission"
        frozen = getattr(sys, "frozen", None)
        try:
            sys.frozen = True
            assert capture.camera_permission_holder() == "Yewee"
        finally:
            if frozen is None:
                del sys.frozen
            else:
                sys.frozen = frozen


@run("camera: devices are chosen by name, not by position in a list")
def _():
    from yewee import capture

    real_names = capture._camera_names
    try:
        # The Blackmagic scenario: the enumeration order flips between two
        # moments (it really does — per process and per replug). A saved
        # index silently becomes a different physical device; a name must
        # either find the right one or refuse loudly.
        capture._camera_names = lambda: ["Blackmagic UltraStudio Recorder 3G",
                                         "FaceTime HD Camera"]
        assert capture.resolve_camera("FaceTime HD Camera") == 1
        assert capture.resolve_camera("blackmagic") == 0, "loose match works"

        capture._camera_names = lambda: ["FaceTime HD Camera",
                                         "Blackmagic UltraStudio Recorder 3G"]
        assert capture.resolve_camera("FaceTime HD Camera") == 0, \
            "same name, new order, still the right device"

        try:
            capture.resolve_camera("DeckLink 8K Pro")
            raise AssertionError("an absent camera must refuse, not guess")
        except RuntimeError as exc:
            assert "connected now" in str(exc), "the error lists what exists"

        # Ambiguity must refuse too — two Blackmagic boxes, 'blackmagic'
        # could be either, and guessing wrong feeds the wrong camera to a
        # live output.
        capture._camera_names = lambda: ["Blackmagic UltraStudio Recorder 3G",
                                         "Blackmagic UltraStudio 4K"]
        try:
            capture.resolve_camera("blackmagic")
            raise AssertionError("ambiguous names must not be guessed")
        except RuntimeError:
            pass
    finally:
        capture._camera_names = real_names

    # The probe hands the panel name specs, so what the user clicks is a
    # device, not a position.
    entries = capture.probe_cameras(max_index=0)
    assert entries == []  # no devices probed, but the call shape holds


@run("onnx: a provider that is listed but cannot load falls back")
def _():
    try:
        import onnxruntime as ort
    except ImportError:
        print("        (onnxruntime not installed — skipped)")
        return
    from yewee import runtime
    from yewee.detectors import CENTERFACE_MODEL

    real_available = ort.get_available_providers
    real_session = ort.InferenceSession
    tried = []

    # Exactly the Windows failure: onnxruntime-gpu advertises CUDA and
    # TensorRT, then loading them dies on a missing cublas64_12.dll.
    def fake_available():
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider",
                *real_available()]

    def fake_session(path, **kw):
        wanted = (kw.get("providers") or ["CPUExecutionProvider"])[0]
        tried.append(wanted)
        if wanted != "CPUExecutionProvider":
            raise RuntimeError(
                "Error loading onnxruntime_providers_tensorrt.dll which "
                "depends on cublas64_12.dll which is missing")
        return real_session(path, **kw)

    ort.get_available_providers, ort.InferenceSession = fake_available, fake_session
    try:
        session = runtime.make_session(CENTERFACE_MODEL,
                                       ["TensorrtExecutionProvider",
                                        "CUDAExecutionProvider",
                                        "CPUExecutionProvider"])
    finally:
        ort.get_available_providers = real_available
        ort.InferenceSession = real_session

    assert tried[0] == "TensorrtExecutionProvider", "should try the best first"
    assert tried[-1] == "CPUExecutionProvider", f"should end on CPU, tried {tried}"
    assert session.get_providers(), "a working session must come back"


@run("licensing: no public key means an unrestricted build")
def _():
    from yewee import licensing as lic
    assert lic.VENDOR_PUBLIC_KEY == "", "repo must not carry a product key"
    st = lic.status()
    assert st["state"] == "unrestricted", "internal builds must not be gated"
    assert not lic.is_blocked(st)
    assert len(lic.machine_id()) == 16


@run("admin tool: issues keys the app accepts, and never ships")
def _():
    import importlib.util
    import secrets as _s
    from datetime import date
    from yewee import _ed25519 as ed, licensing as lic

    spec = importlib.util.spec_from_file_location(
        "ft_admin", os.path.join(ROOT, "tools", "admin.py"))
    admin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admin)          # imports without starting a server

    # the vendor app must be able to build a key the product will accept
    secret = _s.token_bytes(32)
    pub = ed.public_key(secret).hex()
    payload = {"v": 1, "p": "yewee", "e": "pro", "n": "Admin Test",
               "i": date.today().isoformat(), "k": "deadbe"}
    key = lic.encode_key(payload, secret)
    assert lic.decode_key(key, public_key_hex=pub)["n"] == "Admin Test"

    # and it must be excluded from anything shipped
    assert "tools" not in os.listdir(os.path.join(ROOT, "yewee")), \
        "vendor tooling must live outside the shipped package"
    assert hasattr(admin, "build_app") and hasattr(admin, "vendor_dir")


# ---- shop licences (letissier.ie) ----------------------------------------
# The vectors are the shop's own (letissier.ie clients/vectors.json, copied
# into tests/), signed with a test key they carry. Every SDK the shop ships
# agrees on them, so Yewee's verifier has to as well.

def _shop_vectors():
    import json
    with open(os.path.join(ROOT, "tests", "letissier_vectors.json")) as f:
        return json.load(f)


class _ShopSandbox:
    """A licensed build with its state in a temp dir: the given signing
    key, a fixed machine, and no trial anchor outside the sandbox."""

    def __init__(self, shop_key_hex, fingerprint="TEST-MACHINE-0001"):
        from yewee import licensing as lic
        self.lic, self.dir = lic, tempfile.TemporaryDirectory()
        self.saved = {n: getattr(lic, n) for n in (
            "VENDOR_PUBLIC_KEY", "SHOP_PUBLIC_KEY", "user_data_dir",
            "_secondary_anchor", "shop_fingerprint", "_post", "_held_this_run")}
        d = Path(self.dir.name)
        lic.VENDOR_PUBLIC_KEY = "00" * 32          # licensing switched on
        lic.SHOP_PUBLIC_KEY = shop_key_hex
        lic.user_data_dir = lambda: d
        lic._secondary_anchor = lambda: d / "anchor"
        lic.shop_fingerprint = lambda: fingerprint
        lic._held_this_run = False
        self.fingerprint = fingerprint

    def restore(self):
        for n, v in self.saved.items():
            setattr(self.lic, n, v)
        self.dir.cleanup()


def _shop_token(secret, **over):
    """Mint a token the way the service does: sign the ASCII of the
    base64url payload segment."""
    import json
    import time as _t
    from yewee import _ed25519 as ed, licensing as lic
    now = int(_t.time())
    claims = {"v": 1, "key": "LT-YEWE-K9HZ-NGVX-PHJB", "product": "yewee",
              "edition": "standard", "customer": "c", "name": "Shop Buyer",
              "seats": 2, "maintUntil": now + 3650 * 86400, "exp": now + 30 * 86400,
              "machine": lic.shop_machine_hash("TEST-MACHINE-0001"),
              "mode": "online", "iat": now, "jti": "j"}
    claims.update(over)
    seg = lic._b64e(json.dumps(claims, separators=(",", ":")).encode())
    return f"{seg}.{lic._b64e(ed.sign(secret, seg.encode()))}"


@run("shop licences: the shop's published vectors agree")
def _():
    from yewee import licensing as lic
    v = _shop_vectors()
    key, fp, t = v["publicKeyHex"], v["fingerprint"], v["tokens"]

    def check(tok, build=v["claims"]["iat"], now=v["now"], machine=fp):
        # the vectors are issued for vizz; Yewee's own product is checked below
        return lic.check_token(tok, machine, build, now, key, product="vizz")[0]

    assert lic.shop_machine_hash(fp) == v["machineHash"]
    assert check(t["valid"]) == lic.ACTIVE
    for bad in ("tampered", "wrongKey", "malformed"):
        assert check(t[bad]) == lic.INVALID, f"{bad} verified"
    for row in v["entitlement"]:
        assert check(t["valid"], build=row["buildDate"]) == row["expect"], row["why"]
    for row in v["lease"]:
        assert check(t["valid"], now=row["at"]) == row["expect"], row["why"]
    for row in v["trialLease"]:
        assert check(t["validTrial"], now=row["at"]) == row["expect"], row["why"]
    assert check(t["valid"], machine="SOME-OTHER-MACHINE") == lic.WRONG_MACHINE
    # the lease lapses AT exp, as in the SDK and Light, not a second later
    exp = v["claims"]["exp"]
    assert check(t["valid"], now=exp - 1) == lic.ACTIVE
    assert check(t["valid"], now=exp) == lic.CHECK_IN_REQUIRED
    # a correctly signed licence for another of the studio's products is not
    # a Yewee licence
    assert lic.check_token(t["valid"], fp, 0, v["now"], key)[0] == lic.INVALID


@run("shop licences: the studio's signing key is compiled in")
def _():
    from yewee import licensing as lic
    assert lic.SHOP_PUBLIC_KEY == \
        "1fca6c21f2eb7963fd646272a731a41a191d3a4cda839e295c5cda67978fcc85"
    assert lic.SHOP_PRODUCT == "yewee"


@run("shop keys: typos and other products are caught before the network")
def _():
    from yewee import licensing as lic
    # real keys, made by the shop's own generateKey
    assert lic.shop_key_problem("LT-YEWE-K9HZ-NGVX-PHJB") == ""
    assert lic.shop_key_problem("lt-yewe-2z66-z1c0-awjv") == ""
    assert lic.shop_key_problem(" YEWE-2Z66-Z1CO-AWJV ") == "", "O read as 0"
    assert lic.normalise_shop_key("1t-yewe-2z66-z1c0-awjv") == "LT-YEWE-2Z66-Z1C0-AWJV"
    assert "typo" in lic.shop_key_problem("LT-YEWE-K9HZ-NGVX-PHJC")
    assert "Light" in lic.shop_key_problem("LT-11GH-2NXS-Q2KF-HTHQ")
    assert "Vizz" in lic.shop_key_problem("LT-V1ZZ-6W2S-40PQ-CNCM")
    assert "doesn't look like" in lic.shop_key_problem("hello@example.com")


@run("shop licences: what each status costs, and YW1 keys still work")
def _():
    import secrets as _s
    import time as _t
    from yewee import _ed25519 as ed
    secret = _s.token_bytes(32)
    box = _ShopSandbox(ed.public_key(secret).hex())
    lic = box.lic
    try:
        def with_token(**over):
            lic._write_shop({"key": "LT-YEWE-K9HZ-NGVX-PHJB",
                             "token": _shop_token(secret, **over)})
            return lic.status()

        st = with_token()
        assert st["state"] == "licensed" and st["source"] == "shop", st
        assert st["name"] == "Shop Buyer" and not st["note"]
        assert not lic.is_blocked(st)

        # a lapsed lease only asks for a check-in
        st = with_token(exp=int(_t.time()) - 60)
        assert st["state"] == "licensed" and "checked in" in st["note"], st

        # an ended update window costs newer builds, never this one
        saved_build = lic.BUILD_DATE
        lic.BUILD_DATE = int(_t.time())
        try:
            st = with_token(maintUntil=int(_t.time()) - 86400)
            assert st["state"] == "licensed" and "updates ran" in st["note"], st
        finally:
            lic.BUILD_DATE = saved_build

        # a token copied from another machine unlocks nothing
        lic._held_this_run = False
        st = with_token(machine=lic.shop_machine_hash("ANOTHER-MACHINE"))
        assert st["state"] == "trial" and "another machine" in st["note"], st

        # a token for another product unlocks nothing
        st = with_token(product="vizz")
        assert st["state"] == "trial", st

        # a shop trial that ran out falls back to the app's own trial clock
        st = with_token(edition="trial", exp=int(_t.time()) - 1)
        assert st["state"] == "trial" and "trial has ended" in st["note"], st

        # without a shop licence, a YW1 key licenses the machine as before
        lic._shop_path().unlink()
        payload = {"v": 1, "p": "yewee", "e": "pro", "n": "Old Buyer",
                   "i": "2026-07-28", "k": "abc123"}
        vendor = _s.token_bytes(32)
        lic.VENDOR_PUBLIC_KEY = ed.public_key(vendor).hex()
        ok, _msg = lic.activate(lic.encode_key(payload, vendor))
        assert ok, _msg
        st = lic.status()
        assert st["state"] == "licensed" and st["source"] == "yw1", st
        assert st["name"] == "Old Buyer"
    finally:
        box.restore()


@run("shop licences: a refund ends the licence at the next launch, not mid-show")
def _():
    import secrets as _s
    from yewee import _ed25519 as ed
    secret = _s.token_bytes(32)
    box = _ShopSandbox(ed.public_key(secret).hex())
    lic = box.lic
    try:
        lic._write_shop({"key": "LT-YEWE-K9HZ-NGVX-PHJB",
                         "token": _shop_token(secret)})
        assert lic.status()["state"] == "licensed"          # the show starts

        def refused(path, body, timeout=15.0):
            raise lic.ShopError("revoked", "This licence has been revoked.")
        lic._post = refused
        assert "ended" in lic.check_in()

        st = lic.status()
        assert st["state"] == "licensed", "a revoked licence must not stop a running show"
        assert "keeps running" in st["note"]

        lic._held_this_run = False                          # the next launch
        st = lic.status()
        assert st["state"] != "licensed" and "revoked" in st["note"], st

        # a network failure is never a licensing failure
        lic._write_shop({"key": "LT-YEWE-K9HZ-NGVX-PHJB",
                         "token": _shop_token(secret)})

        def offline(path, body, timeout=15.0):
            raise lic.ShopError("network", "Couldn't reach letissier.ie.")
        lic._post = offline
        assert "keeping the stored licence" in lic.check_in()
        lic._held_this_run = False
        assert lic.status()["state"] == "licensed"
    finally:
        box.restore()


@run("shop licences: activation, check-in and release over the wire")
def _():
    import secrets as _s
    from yewee import _ed25519 as ed
    secret = _s.token_bytes(32)
    box = _ShopSandbox(ed.public_key(secret).hex())
    lic = box.lic
    calls = []
    try:
        def service(path, body, timeout=15.0):
            calls.append((path, body))
            if path == "/api/licence/deactivate":
                return {"ok": True}
            return {"ok": True, "token": _shop_token(secret),
                    "machine": lic.shop_machine_hash(body["machine"])}
        lic._post = service

        ok, msg = lic.activate(" lt-yewe-k9hz-ngvx-phjb ")
        assert ok and "Shop Buyer" in msg, msg
        path, body = calls[-1]
        assert path == "/api/licence/activate"
        assert body["key"] == "LT-YEWE-K9HZ-NGVX-PHJB", "key sent in canonical form"
        # the raw fingerprint goes on the wire; the service hashes it
        assert body["machine"] == box.fingerprint
        # the service refuses another product's key before taking a seat
        # only when told which product is asking
        assert body["product"] == "yewee", body
        assert lic.status()["state"] == "licensed"

        assert lic.check_in() == "checked in"
        assert calls[-1][0] == "/api/licence/heartbeat"
        assert calls[-1][1] == {"key": "LT-YEWE-K9HZ-NGVX-PHJB",
                                "machine": box.fingerprint}

        msg = lic.deactivate()
        assert calls[-1][0] == "/api/licence/deactivate" and "released" in msg
        assert not lic._shop_path().exists()
        assert lic.status()["state"] != "licensed"

        # a service that recorded a different machine: refuse, store nothing
        lic._post = lambda p, b, timeout=15.0: {
            "ok": True, "token": _shop_token(secret), "machine": "0" * 32}
        ok, msg = lic.activate("LT-YEWE-K9HZ-NGVX-PHJB")
        assert not ok and "Nothing was stored" in msg, msg
        assert not lic._shop_path().exists()

        # no network: say how to activate offline, with the request code
        def offline(path, body, timeout=15.0):
            raise lic.ShopError("network", "Couldn't reach letissier.ie.")
        lic._post = offline
        ok, msg = lic.activate("LT-YEWE-K9HZ-NGVX-PHJB")
        assert not ok and box.fingerprint in msg and "offline" in msg, msg

        # a typo never reaches the service
        ok, msg = lic.activate("LT-YEWE-K9HZ-NGVX-PHJC")
        assert not ok and "typo" in msg

        # the service's own refusal is passed on as it words it
        def no_seats(path, body, timeout=15.0):
            raise lic.ShopError("no_seats", "All 2 seats are in use.")
        lic._post = no_seats
        assert lic.activate("LT-YEWE-K9HZ-NGVX-PHJB") == (False, "All 2 seats are in use.")

        # offline activation: the licence pasted from the account page
        ok, msg = lic.activate(_shop_token(secret, mode="offline"))
        assert ok, msg
        assert lic._read_shop()["key"] == "LT-YEWE-K9HZ-NGVX-PHJB", "key kept for check-ins"
        ok, msg = lic.activate(_shop_token(secret,
                               machine=lic.shop_machine_hash("typed-differently")))
        assert not ok and box.fingerprint in msg, "say what the request code should be"
    finally:
        box.restore()


@run("edition: GPL-only models are excluded from distribution builds")
def _():
    import importlib
    from yewee import edition, segmenter

    internal = {m["value"] for m in segmenter.available_people_models()}
    assert "rvm" in internal, "internal builds keep RVM (it is not distributed)"

    orig = edition.DISTRIBUTION
    try:
        edition.DISTRIBUTION = True
        importlib.reload(segmenter)  # picks the flag up through _usable()
        shipped = {m["value"] for m in segmenter.available_people_models()}
        assert "rvm" not in shipped, "GPL-3.0 model must not ship in a sold build"
        assert "modnet" in shipped and "pphumanseg" in shipped
        try:
            segmenter.create_people_model("rvm")
            raise AssertionError("distribution build must refuse to load RVM")
        except RuntimeError as exc:
            assert "GPL-3.0" in str(exc)
    finally:
        edition.DISTRIBUTION = orig
        importlib.reload(segmenter)


@run("detector: no non-distributable model is referenced")
def _():
    from yewee import doctor
    from yewee.detectors import CenterFaceDetector, YuNetDetector  # noqa: F401
    from yewee.params import SPEC
    assert "scrfd" not in SPEC["detector"][1], "SCRFD is non-commercial; must be gone"
    assert not any("scrfd" in name for name in doctor.MODELS)


@run("runtime: CPU limit caps OpenCV and ONNX threads")
def _():
    import cv2 as _cv
    from yewee import runtime
    default = _cv.getNumThreads()
    try:
        n = runtime.limit_threads(True)
        c = runtime.cores()
        # assert the policy's intent, not its arithmetic
        assert 1 <= n <= 6, f"budget {n} outside 1..6 on {c} cores"
        if c >= 3:
            assert n < c, f"must leave cores in reserve ({n} of {c})"
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
    from yewee.params import LiveParams
    from yewee.pipeline import Pipeline

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
    from yewee.overlay import render_test_card
    card, ovl = render_test_card(640, 360, ["yewee TEST CARD", "MAC", "640x360"])
    assert card.shape == (360, 640, 3) and ovl.shape == (360, 640, 4)
    bar_row = card[10]
    uniques = len(np.unique(bar_row.reshape(-1, 3), axis=0))
    assert uniques >= 7, f"expected 7 colour bars, saw {uniques} colours"
    ramp_row = card[int(360 * 0.65)]
    assert ramp_row[..., 0].max() - ramp_row[..., 0].min() > 200, "ramp missing"
    alpha = ovl[:, :, 3]
    assert 0.001 < (alpha > 0).mean() < 0.2, "alpha test pattern coverage odd"
    assert ovl[alpha == 0].max() == 0, "alpha card must be empty where transparent"


@run("outputs: flip is per-transport and never mutates the shared frame")
def _():
    from yewee.texture_out import SpoutOutput, SyphonOutput

    # Both transports must start the right way up, and each carries its own
    # setting — a shared one would fix whichever feed was wrong and invert
    # the other. (Texture share flips by inverting its publish flag, which
    # needs a live Syphon/Spout server to exercise; the default is what can
    # be checked anywhere.)
    for cls in (SyphonOutput, SpoutOutput):
        assert cls.flip is False, f"{cls.__name__} must default to no flip"

    try:
        from yewee.ndi_io import NDIOutput
    except ImportError:
        print("        (cyndilib not installed — NDI half skipped)")
        return
    assert NDIOutput.flip is False, "NDI must default to no flip"

    # One frame is shared by every feed, so flipping must produce a new
    # array. In place, it would turn the picture over for the other
    # transport too — and again for each feed that reuses the frame.
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[0, :] = 255                       # a bright top row to follow
    original = frame.copy()

    sent = {}
    out = NDIOutput.__new__(NDIOutput)      # no NDI runtime needed here
    out.size, out._hold = (8, 8), None
    out._reopen = lambda w, h: None
    out.sender = type("S", (), {"write_video_async": lambda self, buf:
                                sent.__setitem__("buf", buf.copy())})()

    out.flip = False
    out.send(frame)
    assert sent["buf"].reshape(8, 8, 4)[0, 0, 0] == 255, "unflipped: top row stays top"

    out.flip = True
    out.send(frame)
    flipped = sent["buf"].reshape(8, 8, 4)
    assert flipped[7, 0, 0] == 255 and flipped[0, 0, 0] == 0, "flipped: top row goes bottom"
    assert np.array_equal(frame, original), "send() must not modify the caller's frame"


@run("params: string choices validate")
def _():
    from yewee.params import LiveParams, SPEC
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
    from yewee.params import LiveParams
    from yewee.pipeline import Pipeline

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


def _quiet_pipeline(**over):
    from main import DEFAULTS, parse_args
    from yewee.params import LiveParams
    from yewee.pipeline import Pipeline
    args = parse_args(["--source", os.path.join(ROOT, "test_media", "synth.mp4"),
                       "--no-ndi", "--no-preview", "--no-web", "--no-browser",
                       "--quiet", "--backend", "yunet", "--loop"])
    params = LiveParams(**{**DEFAULTS, "ndi_program": False, "panel_preview": False,
                           "local_preview": False, "emotion_enabled": False, **over})
    return Pipeline(args, params, web_enabled=False), params


@run("pipeline: a frame that raises is skipped and tracking carries on")
def _():
    import threading
    import time
    pipe, _params = _quiet_pipeline(detector="yunet")
    real = pipe.detector.detect
    calls = {"n": 0}

    def sometimes_broken(frame):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("simulated detector fault")
        return real(frame)

    pipe.detector.detect = sometimes_broken
    reported = []
    pipe.on_frame_error = reported.append
    pipe.args.max_frames = 12
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        done = pipe.run()
    assert done == 12, f"loop ended early after {done} good frames"
    assert calls["n"] >= 17, calls
    assert len(reported) == 1, "each distinct failure is reported once per run"
    assert pipe.get_stats().get("frame_errors", 0) >= 5

    # every frame failing: the loop stays up, outputs fall back to black,
    # and it stops cleanly when asked
    pipe2, params2 = _quiet_pipeline(detector="yunet")   # no engine swap mid-test

    def always_broken(frame):
        raise ValueError("every frame is bad")

    pipe2.detector.detect = always_broken
    sent = []

    class Out:
        flip = False

        def __init__(self, name, fps=30.0):
            pass

        def send(self, img):
            sent.append(int(img.max()))

        def close(self):
            pass

    # stand in for NDI (absent in CI) with an output that records frames
    import types
    fake_ndi = types.ModuleType("yewee.ndi_io")
    fake_ndi.NDIOutput = Out
    saved_mod = sys.modules.get("yewee.ndi_io")
    sys.modules["yewee.ndi_io"] = fake_ndi
    params2.set("ndi_program", True)
    t = threading.Thread(target=pipe2.run, daemon=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            t.start()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and not sent:
                time.sleep(0.05)
            alive = t.is_alive()
            pipe2.stop()
            t.join(timeout=5)
    finally:
        if saved_mod is not None:
            sys.modules["yewee.ndi_io"] = saved_mod
        else:
            sys.modules.pop("yewee.ndi_io", None)
    assert alive, "the loop died instead of skipping failed frames"
    assert sent and max(sent) == 0, "persistent failures must put black on the feeds"
    assert "tracking continues" in pipe2.get_stats().get("error", "")


@run("pipeline: a saved detector that no longer loads falls back at startup")
def _():
    from yewee import pipeline as pl
    real = pl.pick_backend

    def picky(backend, size, threshold):
        if backend != "yunet":
            raise RuntimeError("no GPU runtime here")
        return real(backend, size, threshold)

    pl.pick_backend = picky
    try:
        pipe, params = _quiet_pipeline(detector="centerface")
    finally:
        pl.pick_backend = real
    try:
        assert pipe.detector.name.startswith("yunet"), pipe.detector.name
        assert params.snapshot()["detector"] == "yunet"
        assert "unavailable" in pipe.startup_error
    finally:
        pipe.source.close()


def _child(code: str, d: str) -> int:
    """Run `code` in a fresh interpreter with the crash guard following it
    in directory d, the way main() does."""
    import subprocess
    prog = ("import sys; sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from yewee import crashguard\n"
            "D = Path(%r)\n"
            "crashguard.begin('1.0.0', D)\n" % (ROOT, d)) + code
    return subprocess.run([sys.executable, "-c", prog], capture_output=True,
                          timeout=60).returncode


@run("crash guard: a clean exit leaves nothing; each kind of death is told apart")
def _():
    from yewee import crashguard as cg
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # clean: begin -> end_clean leaves no marker and no report
        assert _child("crashguard.end_clean(D)\n", td) == 0
        assert not list(d.glob("running-*")), "a clean exit must remove the marker"
        assert cg.previous_session(d) is None

        # an uncaught exception on the main thread
        assert _child("raise ValueError('boom at frame 12')\n", td) != 0
        rep = cg.previous_session(d)
        assert rep and rep["kind"] == "exception", rep
        assert rep["summary"] == "ValueError: boom at frame 12", rep
        assert "Traceback" in rep["detail"] and rep["version"] == "1.0.0"
        assert cg.previous_session(d) is None, "a report is produced once"

        # a native crash no Python hook can see: faulthandler's stack
        if sys.platform != "win32":
            assert _child("import faulthandler; faulthandler._sigsegv()\n", td) != 0
            rep = cg.previous_session(d)
            assert rep and rep["kind"] == "signal", rep
            assert "Segmentation fault" in rep["summary"], rep

        # killed outright (power cut, force quit): only the marker is left
        assert _child("import os; os._exit(9)\n", td) == 9
        rep = cg.previous_session(d)
        assert rep and rep["kind"] == "unclean-exit" and rep["detail"] == "", rep

        # the watchdog's hang dump
        assert _child("crashguard.record_hang('stalled for 30s', D)\n"
                      "import os; os._exit(3)\n", td) == 3
        rep = cg.previous_session(d)
        assert rep and rep["kind"] == "hang" and "stalled" in rep["summary"], rep
        assert "Thread" in rep["detail"] or "File" in rep["detail"], rep["detail"]

        # a marker whose process is still alive is another instance, not a crash
        other = os.getppid()
        (d / f"running-{other}.json").write_text('{"version": "1.0.0", "t": 1}')
        assert cg.previous_session(d) is None
        assert (d / f"running-{other}.json").exists(), "a live instance's marker is left"


class _ReportsSandbox:
    """Settings, queue and prompt in a temp dir, and a fake transport that
    records what would have gone over the wire."""

    def __init__(self, statuses=None):
        from yewee import reporting, settings
        self.r, self.settings = reporting, settings
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.saved = (settings.SETTINGS_PATH, reporting.state_dir, reporting._send)
        settings.SETTINGS_PATH = d / "settings.json"
        reporting.state_dir = lambda: d
        reporting._reset_for_tests()
        self.sent: list[tuple[str, dict]] = []
        self.statuses = list(statuses or [])

        def fake_send(endpoint, payload):
            self.sent.append((endpoint, payload))
            status = self.statuses.pop(0) if self.statuses else 202
            if status is None:
                raise OSError("offline")
            return status

        reporting._send = fake_send

    def restore(self):
        self.settings.SETTINGS_PATH, self.r.state_dir, self.r._send = self.saved
        self.r._reset_for_tests()
        self.dir.cleanup()


@run("reports: the scrubber keeps paths, keys, names and addresses on the machine")
def _():
    from yewee import reporting as r
    home = str(Path.home())
    r._reset_for_tests()
    r.add_secrets("ndi:STUDIO-PC (PTZ Cam 1)", "cam:Blackmagic UltraStudio", "0")
    text = (f'File "{home}/shows/yewee/main.py", line 3, in main\n'
            'File "/Users/jane/Movies/crowd.mov"\n'
            'File "C:\\Users\\Jane Doe\\AppData\\yewee\\x.py"\n'
            "GET https://api.example.com/v1/x?token=abc123&user=jane\n"
            "key LT-YEWE-K9HZ-NGVX-PHJB and YW1.eyJhIjoxfQ.c2ln\n"
            "mail jane@example.org from 192.168.1.20 via fe80:0:0:0:1:2:3:4\n"
            "source 'STUDIO-PC (PTZ Cam 1)' and camera Blackmagic UltraStudio\n"
            "NDI source matching 'x' not found. Visible sources: ['A (B)', 'C (D)']\n"
            "bound on 127.0.0.1 with macos arm64")
    out = r.scrub(text)
    for gone in (home, "jane", "Jane Doe", "token=abc123", "K9HZ", "YW1.eyJ",
                 "jane@example.org", "192.168.1.20", "fe80:", "PTZ Cam 1",
                 "UltraStudio", "A (B)"):
        assert gone.lower() not in out.lower(), f"{gone!r} survived:\n{out}"
    for kept in ("~/shows/yewee/main.py", "/Users/<user>/Movies",
                 "https://api.example.com/v1/x", "127.0.0.1", "macos arm64",
                 "<licence>", "<email>", "<ip>"):
        assert kept in out, f"{kept!r} missing:\n{out}"
    r._reset_for_tests()


@run("reports: payloads keep to the contract's fields and limits")
def _():
    box = _ReportsSandbox()
    try:
        r = box.r
        huge = "x" * 100_000
        p = r.crash_payload("panic-ish", "first line\nValueError: " + huge,
                            'File "a.py", line 1, in f\n' + huge,
                            note="n" * 5000, occurred_at="2026-09-24T02:10:00Z")
        assert set(p) <= set(r.CRASH_FIELDS), set(p) - set(r.CRASH_FIELDS)
        assert p["product"] == "yewee" and p["version"] == "1.0.0"
        assert p["kind"] == "other", "unknown kinds become other"
        assert p["os"] in ("macos", "windows", "linux")
        assert len(p["summary"]) <= 300 and len(p["detail"]) <= 32768
        assert len(p["note"]) <= 2000 and len(p["signature"]) <= 128
        assert len(json.dumps(p).encode()) <= r.MAX_BODY
        assert re.fullmatch(r"[A-Za-z0-9-]{8,64}", p["install"])
        assert p["install"] == r.install_id(), "one id per install"
        raw = json.loads(box.settings.SETTINGS_PATH.read_text())
        assert raw["reports"]["install"] == p["install"], "kept with the settings"
        # the same crash from another machine or build has the same signature
        a = r.signature("exception", "KeyError: 'x'",
                        'File "/Users/a/yewee/pipeline.py", line 10, in run\n')
        b = r.signature("exception", "KeyError: 'y'",
                        'File "C:\\\\x\\\\yewee\\\\pipeline.py", line 99, in run\n')
        assert a == b, "signature ignores paths, line numbers and message"
        # a crash never carries a licence, e-mail or name, even if typed
        p = r.crash_payload("exception", "LT-YEWE-K9HZ-NGVX-PHJB jane@x.org", "",
                            note="my key is LT-YEWE-K9HZ-NGVX-PHJB")
        blob = json.dumps(p)
        assert "K9HZ" not in blob and "jane@x.org" not in blob
        assert not ({"licence", "email", "name"} & set(p))

        f = r.feedback_payload("idea", "  map a MIDI fader  ")
        assert f["message"] == "map a MIDI fader" and f["public"] is False
        assert "licence" not in f and "email" not in f
        assert set(f) <= set(r.FEEDBACK_FIELDS)
        f = r.feedback_payload("bug", "x", email="me@example.com",
                               licence="LT-YEWE-K9HZ-NGVX-PHJB", public=True)
        assert f["email"] == "me@example.com" and f["public"] is True
        assert f["licence"] == "LT-YEWE-K9HZ-NGVX-PHJB"
        for bad in (("nope", "x"), ("bug", "  "), ("bug", "y" * 5001)):
            try:
                r.feedback_payload(*bad)
                raise AssertionError(f"accepted {bad[0]!r}/{len(bad[1])} chars")
            except ValueError:
                pass
        try:
            r.feedback_payload("bug", "x", email="not an address")
            raise AssertionError("accepted a bad e-mail")
        except ValueError:
            pass
    finally:
        box.restore()


@run("reports: the queue keeps 20, drops what the service refuses, waits when offline")
def _():
    box = _ReportsSandbox()
    try:
        r = box.r
        for i in range(25):
            r.enqueue("crash", r.crash_payload("exception", f"E{i}: x"))
        items = r.queued()
        assert len(items) == 20, len(items)
        first = json.loads(items[0].read_text())["payload"]["summary"]
        assert first == "E5: x", f"oldest must be dropped first, head is {first}"
        # offline: nothing is lost, and it stops at the first failure
        box.statuses = [None]
        got = r.flush()
        assert got["sent"] == 0 and len(r.queued()) == 20 and len(box.sent) == 1
        # stored, refused, too large, rate limited: 202/400/413 go, 429 stays
        box.sent.clear()
        box.statuses = [202, 400, 413, 429]
        got = r.flush()
        assert (got["sent"], got["dropped"]) == (1, 2), got
        assert len(r.queued()) == 17 and len(box.sent) == 4
        box.statuses = []
        assert r.flush()["sent"] == 17 and not r.queued()
    finally:
        box.restore()


@run("reports: setting off and no click means nothing is sent")
def _():
    box = _ReportsSandbox()
    try:
        r = box.r
        report = {"kind": "exception", "summary": "RuntimeError: boom",
                  "detail": "Traceback...", "occurredAt": "2026-09-24T02:10:00Z",
                  "version": "1.0.0"}
        assert r.auto_send() is False, "sending is off until switched on"
        assert r.handle_previous_session(report) == "prompt"
        assert r.note_nonfatal(ValueError("skipped frame")) is False
        r.flush()
        r.flush_in_background().join(5)
        assert box.sent == [] and r.queued() == [], "nothing may leave without consent"
        # the question survives a restart until it is answered
        r._reset_for_tests()
        assert r.panel_state()["prompt"]["summary"] == "RuntimeError: boom"
        assert r.handle_previous_session(None) == "prompt"
        assert r.answer_prompt(send=False) == "Not sent."
        assert r.pending_prompt() is None and box.sent == []

        # a click on Send (with a note) sends that one report, once
        r.handle_previous_session(report)
        msg = r.answer_prompt(send=True, note="switching sources on jane@x.org")
        assert msg.startswith("Sent"), msg
        assert len(box.sent) == 1 and box.sent[0][0] == "crash"
        assert box.sent[0][1]["note"] == "switching sources on <email>"
        assert r.auto_send() is False, "one Send is not a standing yes"

        # "Always send": the next crash, and recovered errors, go by themselves
        r.handle_previous_session(report)
        r.answer_prompt(send=True, always=True)
        assert r.auto_send() is True
        box.sent.clear()
        assert r.handle_previous_session(report) == "queued"
        assert r.note_nonfatal(ValueError("skipped frame")) is True
        assert r.note_nonfatal(ValueError("skipped frame")) is False, "once per run"
        r.flush()
        assert [e for e, _ in box.sent] == ["crash", "crash"], box.sent
        # feedback goes only through submit_feedback (the Send button)
        ok, msg = r.submit_feedback("idea", "more presets", public=False)
        assert ok and box.sent[-1][0] == "feedback" and "licence" not in box.sent[-1][1]
    finally:
        box.restore()


@run("reports: the real transport posts JSON to LETISSIER_API with an 8 s timeout")
def _():
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.path, dict(self.headers), json.loads(body)))
            code = 202 if self.path.endswith("/crash") else 429
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true, "id": "r1"}')

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    from yewee import reporting as r
    saved = (r.API_BASE, r._send)
    r.API_BASE, r._send = base, r._http_post
    try:
        assert r.TIMEOUT == 8.0
        payload = {"product": "yewee", "version": "1.0.0", "os": "linux",
                   "kind": "exception", "summary": "x"}
        assert r._http_post("crash", payload) == 202
        assert r._http_post("feedback", {"product": "yewee"}) == 429
        path, headers, body = got[0]
        assert path == "/api/reports/crash" and body == payload
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"].startswith("Yewee/1.0.0 ("), headers["User-Agent"]
        assert got[1][0] == "/api/reports/feedback"
    finally:
        r.API_BASE, r._send = saved
        srv.shutdown()
    # the environment override is read when the module loads
    out = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); "
         "from yewee import reporting; print(reporting.API_BASE)" % ROOT],
        capture_output=True, text=True, env={**os.environ,
                                             "LETISSIER_API": "http://127.0.0.1:9/"})
    assert out.stdout.strip() == "http://127.0.0.1:9", out.stdout + out.stderr
    out = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); "
         "from yewee import reporting; print(reporting.API_BASE)" % ROOT],
        capture_output=True, text=True,
        env={k: v for k, v in os.environ.items() if k != "LETISSIER_API"})
    assert out.stdout.strip() == "https://letissier.ie", out.stdout + out.stderr


@run("notices: every shipped package's licence, the NDI notice, the models")
def _():
    import importlib.metadata as md
    from yewee import notices
    text = notices.generate()
    assert "NDI® is a registered trademark of Vizrt NDI AB" in text
    assert "Copyright (C) 2023-2024 Vizrt NDI AB" in text
    for model in ("face_detection_yunet_2023mar.onnx", "centerface_dynamic.onnx",
                  "emotion-ferplus-8.onnx", "modnet_portrait.onnx",
                  "human_segmentation_pphumanseg_2023mar.onnx"):
        assert model in text, model
        assert os.path.exists(os.path.join(ROOT, "models", model)), model
    assert "numpy" in text and "opencv-python" in text and "Python " in text
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in text.upper() or \
        "LESSER GENERAL PUBLIC" in text.upper(), "FFmpeg's LGPL text must travel"
    installed = {d.metadata["Name"].lower() for d in md.distributions()}
    if "cyndilib" in installed:
        assert "Processing.NDI" in text or "libndi" in text or "NDI SDK license" in text
    listed = text.split("Licence texts", 1)[0]
    assert "\n  pip " not in listed and "pyinstaller" not in listed.lower(), \
        "build-only tools do not ship"
    with open(os.path.join(ROOT, "build", "yewee.spec"), encoding="utf-8") as f:
        spec = f.read()
    assert "THIRD-PARTY-NOTICES.txt" in spec and "NSLocalNetworkUsageDescription" in spec


@run("build: notices and buildinfo are UTF-8 whatever the console's encoding")
def _():
    # The Windows release build died reading the notices child's cp1252
    # output as UTF-8. Here the child is handed Windows' pipe encoding and
    # the build still has to come back with the real dash; and every file
    # the build writes must name its encoding (warn_default_encoding turns
    # a bare write_text into an error on any platform, not just Windows).
    import subprocess
    probe = (
        "import importlib.util, os, sys, tempfile\n"
        "from pathlib import Path\n"
        "spec = importlib.util.spec_from_file_location('ft_build', sys.argv[1])\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "text = b.third_party_notices()\n"
        "assert text.startswith('Yewee \\u2014 third-party notices'), repr(text[:40])\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    b.BUILDINFO = Path(td) / '_buildinfo.py'\n"
        "    b.write_buildinfo(True, 'ab' * 32, '1.0.0')\n"
        "    raw = b.BUILDINFO.read_bytes()\n"
        "    compile(raw, '_buildinfo.py', 'exec')\n"
        "    assert '\\u2014'.encode('utf-8') in raw, raw[:60]\n"
        "print('ok')\n")
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    res = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning",
         "-c", probe, os.path.join(ROOT, "build", "build.py")],
        cwd=ROOT, env=env, capture_output=True, timeout=120)
    assert res.returncode == 0 and res.stdout.strip() == b"ok", \
        res.stderr.decode("utf-8", "replace")[-2000:]


@run("emotion: FER+ labels a face")
def _():
    from yewee.detectors import YuNetDetector
    from yewee.emotion import EMOTIONS, EmotionEstimator
    from yewee.tracker import FaceTracker
    frame = _first_frame()
    trk = FaceTracker(min_hits=1)
    tracks = trk.step(YuNetDetector(score_threshold=0.4).detect(frame))
    est = EmotionEstimator(budget_per_frame=2)
    est.update(frame, tracks, 100)
    labelled = [t for t in tracks if t.emotion is not None]
    assert labelled, "no track got an emotion label"
    assert all(t.emotion[0] in EMOTIONS for t in labelled)


def _load_build_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yewee_build_script", os.path.join(ROOT, "build", "build.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@run("version: declared once as 1.0.0, and a tag that disagrees cannot build")
def _():
    import contextlib
    import io
    import yewee
    assert yewee.__version__ == "1.0.0", yewee.__version__
    assert not os.path.exists(os.path.join(ROOT, "yewee", "_buildinfo.py"))
    assert yewee.app_version() == "1.0.0", "a source run reports __version__"
    build = _load_build_script()
    assert build.source_version() == yewee.__version__
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert build.main(["--print-version"]) == 0
    assert out.getvalue().strip() == "1.0.0"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert build.main(["--print-version", "--version", "v1.0.0"]) == 0
    assert out.getvalue().strip() == "1.0.0", "a tag name is accepted as-is"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for tag in ("1.5", "1.0", "0.0.0"):
            assert build.main(["--distribution", "--version", tag]) == 1, tag
    assert "does not match __version__" in err.getvalue()


@run("expressions: off on a fresh install, a saved choice is kept")
def _():
    from main import build_params, parse_args
    args = parse_args([])
    assert build_params(args, {}).snapshot()["emotion_enabled"] is False
    assert build_params(args, {"emotion_enabled": True}).snapshot()["emotion_enabled"] is True
    no = parse_args(["--no-emotion"])
    assert build_params(no, {"emotion_enabled": True}).snapshot()["emotion_enabled"] is False


@run("--no-ndi: every NDI feed starts off, texture share is left alone")
def _():
    from main import build_params, parse_args
    saved = {f"ndi_{c}": True for c in ("program", "overlay", "faces", "mask")}
    saved.update(tex_program=True, tex_mask=True)
    p = build_params(parse_args(["--no-ndi"]), saved).snapshot()
    on = [c for c in ("program", "overlay", "faces", "mask") if p[f"ndi_{c}"]]
    assert not on, f"--no-ndi left these NDI feeds on: {on}"
    assert p["tex_program"] and p["tex_mask"], "--no-ndi must not touch texture share"
    p = build_params(parse_args([]), saved).snapshot()
    assert all(p[f"ndi_{c}"] for c in ("program", "overlay", "faces", "mask"))


@run("presets: a fresh install's detection settings are the Mid crowd preset")
def _():
    from main import DEFAULTS
    html = open(os.path.join(ROOT, "yewee", "static", "index.html"), encoding="utf-8").read()
    body = re.search(r"\bmid:\s*\{([^}]*)\}", html).group(1)
    mid = {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", body)}
    assert mid, "no Mid crowd preset found in the panel"
    for k, v in mid.items():
        assert abs(DEFAULTS[k] - v) < 1e-9, (
            f"fresh install has {k}={DEFAULTS[k]}, Mid crowd (\"good default\") has {v}")


def _fake_cyndilib():
    """Just enough of cyndilib to import yewee.ndi_io where it isn't
    installed (CI); the tests build NDIInput around fakes anyway."""
    import types
    try:
        import cyndilib  # noqa: F401
        return
    except ImportError:
        pass
    names = ("cyndilib", "cyndilib.sender", "cyndilib.video_frame",
             "cyndilib.wrapper", "cyndilib.wrapper.ndi_structs")
    for n in names:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["cyndilib.sender"].Sender = object
    sys.modules["cyndilib.video_frame"].VideoSendFrame = object
    sys.modules["cyndilib.wrapper.ndi_structs"].FourCC = object


class _FakeNDI:
    """A receiver and its frame sync. Like NDI's, the frame sync keeps
    handing back the last frame after the sender stops."""

    def __init__(self):
        self.connected, self.sending, self.ts, self.stamps = True, True, 0.0, True
        self.xres, self.yres = 8, 4
        self.frame_sync = self
        self.receiver = self

    def is_connected(self):
        return self.connected

    def capture_video(self):
        if self.sending and self.stamps:
            self.ts += 1 / 30

    def get_timestamp_posix(self):
        return self.ts if self.stamps else 9.2e9   # "undefined": one constant

    def __array__(self, dtype=None, copy=None):
        return np.full(self.yres * self.xres * 4, 90, np.uint8)


def _ndi_input(fake):
    _fake_cyndilib()
    from yewee.ndi_io import NDIInput
    inp = NDIInput.__new__(NDIInput)          # skip the network search
    inp.receiver, inp.video_frame = fake, fake
    inp._last_ts, inp._ts_at, inp._ts_changes = None, 0.0, 0
    inp.STALE_AFTER = 0.3
    return inp


@run("NDI input: a sender that stops or goes away reads as lost, not a frozen frame")
def _():
    import time
    fake = _FakeNDI()
    inp = _ndi_input(fake)
    for _ in range(5):
        ok, frame = inp.read(timeout=0.5)
        assert ok and frame.shape == (4, 8, 3)
        time.sleep(0.01)
    # the sender stops sending but stays connected: the frame sync repeats
    # the last picture, which must not count as a live input for long
    fake.sending = False
    t0 = time.monotonic()
    while inp.read(timeout=0.1)[0]:
        assert time.monotonic() - t0 < 2.0, "a repeated frame kept reading as live"
    assert inp.read(timeout=0.1) == (False, None)
    fake.sending = True
    assert inp.read(timeout=0.5)[0], "a sender that comes back reads again"
    # the sender closes: the connection drops
    fake.connected = False
    assert inp.read(timeout=0.2) == (False, None), "a disconnected sender still read as live"

    # a sender that doesn't stamp its frames: only the connection counts
    quiet = _FakeNDI()
    quiet.stamps = False
    inp = _ndi_input(quiet)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.6:
        assert inp.read(timeout=0.2)[0], "an unstamped live sender was dropped"
    quiet.connected = False
    assert inp.read(timeout=0.2) == (False, None)


@run("watchdog: the installed app restarts itself; its Restart relaunches cleanly")
def _():
    import main
    saved = (sys.argv, sys.executable, getattr(sys, "frozen", None))
    try:
        # a packaged app is its own executable, and argv[0] is it again
        sys.frozen, sys.executable = True, "/Applications/Yewee.app/Contents/MacOS/Yewee"
        sys.argv = [sys.executable, "--no-browser", "--wait-for-pid", "41"]
        argv = main._relaunch_argv(["--wait-for-pid", "42"])
        assert argv == [sys.executable, "--no-browser", "--wait-for-pid", "42"], argv
        assert main.parse_args(argv[1:]).wait_for_pid == 42
        # from source: the interpreter, then the script and its options
        del sys.frozen
        sys.executable, sys.argv = "/usr/bin/python3", ["main.py", "--no-browser"]
        assert main._relaunch_argv() == ["/usr/bin/python3", "main.py", "--no-browser"]
    finally:
        sys.argv, sys.executable = saved[0], saved[1]
        if saved[2] is None:
            sys.__dict__.pop("frozen", None)
        else:
            sys.frozen = saved[2]

    # a stall: the installed app relaunches before it exits; from source the
    # exit alone is right, since the launcher's loop starts the next one
    import threading
    import time
    import types
    for frozen in (True, False):
        calls, done = [], threading.Event()

        def fake_exit(code):
            calls.append(("exit", code))
            done.set()
            raise SystemExit

        pipe = types.SimpleNamespace(stopped=False, heartbeat=time.monotonic() - 60)
        real = (main._relaunch_after_stall, main.os, main.time)
        main._relaunch_after_stall = lambda: calls.append(("relaunch",))
        main.os = types.SimpleNamespace(_exit=fake_exit, getpid=os.getpid)
        main.time = types.SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic)
        import yewee.paths as paths
        real_frozen = paths.is_frozen
        paths.is_frozen = lambda: frozen
        from yewee import crashguard
        real_hang = crashguard.record_hang
        crashguard.record_hang = lambda reason, d=None: None
        try:
            main._start_watchdog(pipe)
            assert done.wait(5), "the watchdog never acted on a 60 s stall"
        finally:
            main._relaunch_after_stall, main.os, main.time = real
            paths.is_frozen = real_frozen
            crashguard.record_hang = real_hang
        want = [("relaunch",), ("exit", 3)] if frozen else [("exit", 3)]
        assert calls == want, f"frozen={frozen}: {calls}"

    # the relaunched copy waits for the stalled one to be gone
    t0 = time.monotonic()
    main._wait_for_exit(2 ** 22 + 12345, timeout=5)   # no such process
    assert time.monotonic() - t0 < 1


@run("launchers: every model they wait for is one the doctor provides")
def _():
    # Both launchers re-run setup until each listed model exists, and after
    # setup they give up if one is still missing. A name the doctor cannot
    # download (SCRFD, removed for its licence) made every fresh checkout
    # stop at "Setup did not complete cleanly".
    import re
    from yewee.doctor import MODELS
    for launcher in ("Yewee Mac.command", "Yewee Windows.bat"):
        with open(os.path.join(ROOT, launcher), encoding="utf-8") as f:
            text = f.read()
        wanted = set(re.findall(r"models[/\\]([\w.-]+\.onnx)", text))
        assert wanted, f"{launcher}: no model checks found"
        unknown = wanted - set(MODELS)
        assert not unknown, f"{launcher} waits for models nothing provides: {unknown}"
        missing = {n for n in wanted if not os.path.exists(os.path.join(ROOT, "models", n))}
        assert not missing, f"{launcher} waits for models not in the repo: {missing}"


@run("panel PIN: made on first run and kept; --pin wins; turning it off is kept")
def _():
    import secrets as _secrets
    from main import parse_args
    from yewee import settings
    assert parse_args([]).pin is None, "no --pin must mean 'use the saved one'"
    with tempfile.TemporaryDirectory() as td:
        old, real = settings.SETTINGS_PATH, _secrets.randbelow
        settings.SETTINGS_PATH = Path(td) / "settings.json"
        drawn = []
        try:
            _secrets.randbelow = lambda n: drawn.append(n) or 7
            pin, origin = settings.panel_pin(None)
            assert (pin, origin) == ("0007", "new"), (pin, origin)
            assert drawn == [10_000], "four digits from secrets, not random"
        finally:
            _secrets.randbelow = real
        try:
            raw = lambda: json.loads(settings.SETTINGS_PATH.read_text())
            assert raw()["pin"] == "0007", "a first-run PIN is saved"
            assert settings.panel_pin(None) == ("0007", "saved"), "and reused"
            for _ in range(20):
                assert re.fullmatch(r"\d{4}", settings.new_pin())
            # a hand-written or older PIN wins over a new one
            settings.SETTINGS_PATH.write_text('{"pin": 4721, "params": {"flip": true}}')
            assert settings.panel_pin(None) == ("4721", "saved")
            # --pin wins, and is kept for the next launch
            assert settings.panel_pin(" 2468 ") == ("2468", "cli")
            assert settings.panel_pin(None) == ("2468", "saved")
            assert raw()["params"] == {"flip": True}, "other settings untouched"
            # turning it off is deliberate and stays off
            assert settings.panel_pin("none") == ("", "off")
            assert raw()["pin"] == "none"
            assert settings.panel_pin(None) == ("", "off"), "off must persist"
            assert settings.panel_pin("OFF") == ("", "off")
            # an empty value is not "off": it is a first run
            settings.SETTINGS_PATH.write_text('{"pin": ""}')
            pin, origin = settings.panel_pin(None)
            assert origin == "new" and re.fullmatch(r"\d{4}", pin)
            # the PIN stays a launch setting, never a panel param
            from yewee.params import SPEC
            assert "pin" not in SPEC
        finally:
            settings.SETTINGS_PATH = old


class _StubPipeline:
    """Just enough of Pipeline for the web app: each socket gets one tick,
    then sees the pipeline stopped, so its loop ends."""

    def __init__(self):
        self._stop_next = False
        self.source_spec = "0"
        self.preview_clients = 0
        self.args = None

    @property
    def stopped(self):
        stop, self._stop_next = self._stop_next, False
        return stop

    def licence(self):
        return {"state": "trial"}

    def get_stats(self):
        self._stop_next = True
        return {"fps": 30}


def _asgi_scope(kind, path, peer, host, origin=None):
    from urllib.parse import urlsplit
    parts = urlsplit(path)
    headers = [(b"host", host.encode())]
    if origin:
        headers.append((b"origin", origin.encode()))
    scope = {"type": kind, "asgi": {"version": "3.0"}, "http_version": "1.1",
             "scheme": "http" if kind == "http" else "ws", "path": parts.path,
             "raw_path": parts.path.encode(), "query_string": parts.query.encode(),
             "root_path": "", "headers": headers, "client": (peer, 50123),
             "server": ("127.0.0.1", 8089)}
    if kind == "http":
        scope["method"] = "GET"
    else:
        scope["subprotocols"] = []
    return scope


def _asgi_get(app, path, peer="192.168.1.50", host="192.168.1.20:8089", origin=None):
    """GET through the app itself, from a chosen peer address: no socket,
    and no test-client dependency (CI installs only what the app needs)."""
    import asyncio
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(_asgi_scope("http", path, peer, host, origin), receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body.decode()


def _asgi_ws(app, peer="192.168.1.50", host="192.168.1.20:8089", origin=None, auth=None):
    """Open /ws from a peer, answer an auth request with `auth` (None = say
    nothing), and return (messages received, close code or None)."""
    import asyncio
    got, closed = [], []

    async def go():
        inbox = asyncio.Queue()
        await inbox.put({"type": "websocket.connect"})

        async def receive():
            return await inbox.get()

        async def send(message):
            if message["type"] == "websocket.send":
                msg = json.loads(message["text"])
                got.append(msg)
                if msg["type"] == "auth_required":
                    if auth is None:
                        await inbox.put({"type": "websocket.disconnect", "code": 1000})
                    else:
                        await inbox.put({"type": "websocket.receive",
                                         "text": json.dumps({"type": "auth", "data": auth})})
            elif message["type"] == "websocket.close":
                closed.append(message.get("code"))

        await asyncio.wait_for(
            app(_asgi_scope("websocket", "/ws", peer, host, origin), receive, send), 10)

    asyncio.run(go())
    return got, (closed[0] if closed else None)


@run("panel PIN: other devices need it, the machine itself does not")
def _():
    try:
        import fastapi  # noqa: F401
    except ImportError:
        print("        (fastapi not installed — skipped)")
        return
    from yewee import webui
    from yewee.params import LiveParams
    from main import DEFAULTS

    def app_with(pin):
        return webui.create_app(_StubPipeline(), LiveParams(**DEFAULTS), pin=pin,
                                phone_url="http://192.168.1.20:8089")

    app = app_with("4821")
    # another device: refused without the PIN or with a wrong one, let in with it
    assert _asgi_get(app, "/logs")[0] == 401
    assert _asgi_get(app, "/logs?pin=1111")[0] == 401
    assert _asgi_get(app, "/logs?pin=4821")[0] == 200
    assert _asgi_get(app, "/sources?x=1&pin=%C3%A9")[0] == 401, "odd input is a 401, not a 500"
    assert _asgi_get(app, "/")[0] == 200, "the page itself loads, to ask for the PIN"
    assert _asgi_get(app, "/notices")[0] == 200
    got, code = _asgi_ws(app)
    assert code == 4001 and [m["type"] for m in got] == ["auth_required"], (got, code)
    got, code = _asgi_ws(app, auth="1234")
    assert code == 4001 and not any(m["type"] == "tick" for m in got), "wrong PIN got in"
    got, _ = _asgi_ws(app, auth="4821")
    ticks = [m for m in got if m["type"] == "tick"]
    assert ticks, got
    assert "phones" not in ticks[0], "the PIN is shown only on the machine itself"

    # the machine itself: no PIN asked, and it is told the PIN to read off
    for peer, host in (("127.0.0.1", "localhost:8089"), ("::1", "[::1]:8089"),
                       ("127.0.0.1", "127.0.0.1:8089")):
        assert _asgi_get(app, "/logs", peer=peer, host=host)[0] == 200, (peer, host)
    got, code = _asgi_ws(app, peer="127.0.0.1", host="localhost:8089",
                         origin="http://localhost:8089")
    assert [m["type"] for m in got] == ["tick"], got
    assert got[0]["phones"] == {"pin": "4821", "url": "http://192.168.1.20:8089"}
    # ...but not a page from elsewhere in that browser, or a rebound DNS name
    got, code = _asgi_ws(app, peer="127.0.0.1", host="localhost:8089",
                         origin="https://evil.example")
    assert code == 4001 and [m["type"] for m in got] == ["auth_required"], got
    assert _asgi_get(app, "/logs", peer="127.0.0.1", host="evil.example:8089")[0] == 401

    # four digits cannot be walked through: five wrong and the address waits
    now = [1000.0]
    app.state.pin_gate._clock = lambda: now[0]
    for guess in ("0000", "0001", "0002", "0003", "0004"):
        assert _asgi_get(app, f"/logs?pin={guess}", peer="192.168.1.66")[0] == 401
    assert _asgi_get(app, "/logs?pin=4821", peer="192.168.1.66")[0] == 429
    got, code = _asgi_ws(app, peer="192.168.1.66", auth="4821")
    assert code == 4002 and got[0]["type"] == "auth_locked", got
    assert _asgi_get(app, "/logs?pin=4821")[0] == 200, "other devices are unaffected"
    assert _asgi_get(app, "/logs", peer="127.0.0.1", host="localhost:8089")[0] == 200
    now[0] += webui.LOCKOUT_SECONDS + 1
    assert _asgi_get(app, "/logs?pin=4821", peer="192.168.1.66")[0] == 200

    # PIN turned off: open to everyone, and the machine's panel says so
    open_app = app_with("")
    assert _asgi_get(open_app, "/logs")[0] == 200
    got, _ = _asgi_ws(open_app)
    assert [m["type"] for m in got] == ["tick"]
    got, _ = _asgi_ws(open_app, peer="127.0.0.1", host="localhost:8089")
    assert got[0]["phones"]["pin"] == ""


@run("reports: the panel PIN never leaves the machine")
def _():
    box = _ReportsSandbox()
    try:
        r = box.r
        r.add_pin("4821", "none", "12")
        text = ("  Panel PIN     : 4821   (other devices ask for it once)\n"
                "GET /logs?x=1&pin=4821 HTTP/1.1\n"
                'settings {"pin": "4821", "source": "0"}\n'
                "ValueError: bad value 4821\n"
                "auth pin=12 failed\n"
                'File "main.py", line 482, in main; None of 14821')
        p = r.crash_payload("exception", text, text, note="pin is 4821")
        blob = json.dumps(p)
        assert "4821" not in blob.replace("14821", ""), blob
        assert "pin=12" not in blob, "a short PIN is still caught in a pin= field"
        assert "line 482" in blob and "14821" in blob and "None" in blob, \
            "numbers that are not the PIN survive"
        # feedback is only what the person typed: no field carries the PIN
        ok, _ = r.submit_feedback("idea", "more presets", public=False)
        assert ok and "4821" not in json.dumps(box.sent[-1][1])
    finally:
        box.restore()


@run("installers: both show LeTissier's terms, with the NDI and LGPL clauses")
def _():
    # The NDI SDK licence (3d) requires distribution under our own terms,
    # which must forbid reverse engineering the NDI part and disclaim for
    # Vizrt, and the LGPL needs its carve-out. The text is quoted from
    # letissier.ie/terms ("Components made by others"); keep it in step.
    raw = open(os.path.join(ROOT, "build", "TERMS.txt"), "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf"), "Inno Setup reads a .txt as UTF-8 only with a BOM"
    terms = raw[3:].decode("utf-8")
    for needed in ("LeTissier Creative Studios Ltd", "https://letissier.ie/terms",
                   "Components made by others", "GNU Lesser General Public License",
                   "you may not modify, reverse engineer, decompile or disassemble "
                   "that NDI software",
                   "Vizrt NDI AB gives you no warranty for it and has no liability",
                   "NDI® is a registered trademark of Vizrt NDI AB",
                   "THIRD-PARTY-NOTICES.txt"):
        assert needed in terms, needed
    with open(os.path.join(ROOT, "build", "yewee.iss"), encoding="utf-8") as f:
        assert re.search(r"(?m)^LicenseFile=TERMS\.txt\s*$", f.read()), "Windows installer"
    with open(os.path.join(ROOT, "build", "sign_macos.sh"), encoding="utf-8") as f:
        sign = f.read()
    assert 'cp "$TERMS" "$STAGE/TERMS.txt"' in sign and '-srcfolder "$STAGE"' in sign, \
        "the DMG must carry TERMS.txt beside the app"


def _chromium() -> str | None:
    """A Chromium or Chrome to lay the panel out in. GitHub's Ubuntu image
    has google-chrome; PW_CHROMIUM points at a Playwright download."""
    import shutil
    for path in (os.environ.get("PW_CHROMIUM"), os.environ.get("YEWEE_CHROMIUM")):
        if path:
            if os.path.isdir(path):
                for sub in ("chrome-linux/chrome", "chrome-linux64/chrome", "chrome"):
                    if os.path.isfile(os.path.join(path, sub)):
                        return os.path.join(path, sub)
            elif os.path.isfile(path):
                return path
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    return None


def _panel_in_chromium(probe: str, head: str = "") -> dict | None:
    """Lays the real panel out in a 390 px frame in headless Chromium, runs
    `probe` (a script that ends by posting a JSON string to its parent) and
    returns what it posted, or None when no Chromium is found. `head` goes
    in before the panel's own scripts run."""
    import subprocess
    chrome = _chromium()
    if chrome is None:
        print("        (no Chromium/Chrome found — skipped; set PW_CHROMIUM)")
        return None
    html = open(os.path.join(ROOT, "yewee", "static", "index.html"), encoding="utf-8").read()
    if head:
        html = html.replace("<head>", "<head>" + head, 1)
    # Headless Chrome will not open a window under 500 px, so the panel is
    # laid out in a 390 px frame, which is a 390 px viewport to its media
    # queries; the frame posts its measurement to the page around it.
    wrapper = """<!doctype html><body style="margin:0">
<iframe src="index.html" style="width:390px;height:844px;border:0"></iframe>
<script>addEventListener("message", (e) => { const pre = document.createElement("pre");
  pre.id = "layout-result"; pre.textContent = e.data; document.body.appendChild(pre); });
</script></body>"""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "index.html").write_text(html.replace("</body>", probe + "</body>"),
                                             encoding="utf-8")
        page = Path(td) / "frame.html"
        page.write_text(wrapper, encoding="utf-8")
        res = subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
             "--no-default-browser-check", f"--user-data-dir={td}/profile",
             "--hide-scrollbars", "--window-size=800,900", "--virtual-time-budget=5000",
             "--dump-dom", page.as_uri()],
            capture_output=True, text=True, timeout=90)
    m = re.search(r'<pre id="layout-result">(.*?)</pre>', res.stdout, re.S)
    assert m, "no measurement from Chromium:\n" + res.stderr[-2000:]
    import html as _html
    return json.loads(_html.unescape(m.group(1)))


@run("panel: fits a 390 px phone with no sideways scroll, nothing cut off")
def _():
    # Everything the live panel can show, filled with long values, then
    # measured: the page's width, and any element sticking out of its card
    # (a card clips, so that is a control cut off rather than a scrollbar).
    probe = """<script>
addEventListener("load", () => setTimeout(() => {
  const $ = (id) => document.getElementById(id);
  document.querySelectorAll("details").forEach((d) => d.open = true);
  for (const id of ["licence-card", "cutout-card", "crash-banner"]) $(id).style.display = "";
  $("src-custom-row").classList.add("show");
  $("fb-form").style.display = "flex";
  $("phones").style.display = "block";
  $("phones").innerHTML = "Phones and other computers: <b>http://192.168.100.200:8089</b>"
    + " · PIN <b>4821</b> — asked for once per device; this machine doesn't need it";
  $("feedinfo").innerHTML = "Input: <b>ndi:STUDIO-PC-LONG-NAME (PTZ Camera 1)</b> · out: <b>1920x1080</b>";
  $("s-faces").textContent = "128"; $("s-fps").textContent = "29.97"; $("s-load").textContent = "100";
  const vw = document.documentElement.clientWidth, out = [];
  for (const el of document.querySelectorAll("body *")) {
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") continue;
    const b = el.getBoundingClientRect();
    if (!b.width) continue;
    const card = el.parentElement && el.parentElement.closest(".card");
    const lim = Math.min(vw, card ? card.getBoundingClientRect().right : vw);
    if (b.right > lim + 1) out.push((el.id || el.tagName.toLowerCase()) + " ends at "
      + Math.round(b.right) + " > " + Math.round(lim));
  }
  parent.postMessage(JSON.stringify({ vw, sw: document.documentElement.scrollWidth, out }), "*");
}, 300));
</script>"""
    r = _panel_in_chromium(probe)
    if r is None:
        return
    assert r["vw"] == 390, f"window came out {r['vw']} px wide, not 390"
    assert r["sw"] <= r["vw"], f"the page scrolls sideways: {r['sw']} px wide in a {r['vw']} px phone"
    assert not r["out"], "cut off at 390 px: " + "; ".join(r["out"][:10])


@run("panel: a live tick draws the cost bars and lights the fresh install's preset")
def _():
    from main import DEFAULTS
    # A stand-in for the app's socket: once the panel connects, one tick as
    # the app sends it, with a fresh install's settings and a cost breakdown.
    tick = {"type": "tick", "params": DEFAULTS,
            "stats": {"state": "live", "fps": 30.0, "faces": 3, "load_pct": 40,
                      "budget_ms": 33.3,
                      "perf": {"detect": 9.0, "track": 0.5, "outputs": 3.0}}}
    head = """<script>
window.WebSocket = class {
  constructor() { setTimeout(() => { this.onopen && this.onopen();
    this.onmessage({ data: %s }); }, 50); }
  send() {} close() {}
};
</script>""" % json.dumps(json.dumps(tick))
    probe = """<script>
addEventListener("load", () => setTimeout(() => {
  const bars = [...document.querySelectorAll("#perf .pb")].map(
    (b) => Math.round(b.getBoundingClientRect().width));
  const lit = [...document.querySelectorAll(".preset.active")].map((b) => b.dataset.preset);
  parent.postMessage(JSON.stringify({ bars, lit }), "*");
}, 400));
</script>"""
    r = _panel_in_chromium(probe, head)
    if r is None:
        return
    assert len(r["bars"]) == 4, f"expected 3 stage bars and a total, got {r['bars']}"
    assert all(w > 0 for w in r["bars"]), f"cost bars drew with no width: {r['bars']}"
    assert r["bars"][0] > r["bars"][1], f"the priciest stage isn't the widest bar: {r['bars']}"
    assert r["lit"] == ["mid"], f"a fresh install lights {r['lit']}, not Mid crowd"


if FAILURES:
    print(f"\n{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("\nAll smoke tests passed.")
