"""Smoke tests — no test framework needed:

    .venv/bin/python -m tests.smoke

Covers params validation, settings persistence, the tracker, the YuNet
detector + overlay rendering on the committed test clip, and FER+
expression estimation. Avoids NDI, cameras, and the web server so it runs
anywhere (CI included).
"""
from __future__ import annotations

import logging
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


if FAILURES:
    print(f"\n{len(FAILURES)} test(s) failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("\nAll smoke tests passed.")
