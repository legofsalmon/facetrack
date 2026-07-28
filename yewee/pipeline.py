"""The processing pipeline as a controllable object.

Runs the capture -> detect -> track -> emotion -> overlay -> NDI loop,
reading LiveParams each frame so the web UI (or anything else) can adjust
settings mid-run. Also publishes stats and a throttled JPEG preview for
the control panel, and supports hot-swapping the input source.
"""
from __future__ import annotations

import socket
import threading
import time

import cv2
import numpy as np

from .capture import NullSource, open_source, parse_cap_format
from .detectors import pick_backend
from .emotion import EmotionEstimator
from . import runtime as _rt
from .overlay import (apply_cutout, cutout_alpha, draw_stats, draw_tracks,
                      hard_rect_regions, render_mask, render_overlay_bgra,
                      render_test_card)
from .params import LiveParams
from .tracker import FaceTracker


def _people_model_choices():
    """Which silhouette engines this build can offer (cached)."""
    global _PEOPLE_CHOICES
    if _PEOPLE_CHOICES is None:
        try:
            from .segmenter import available_people_models
            _PEOPLE_CHOICES = available_people_models()
        except Exception:
            _PEOPLE_CHOICES = []
    return _PEOPLE_CHOICES


_PEOPLE_CHOICES = None
PREVIEW_INTERVAL = 0.08   # ~12 fps JPEG preview for the panel
PREVIEW_WIDTH = 640


class Pipeline:
    def __init__(self, args, params: LiveParams, web_enabled: bool = False):
        self.args = args
        self.params = params
        self.web_enabled = web_enabled

        p = params.snapshot()
        self._detector_choice = p["detector"]
        self.detector = pick_backend(p["detector"], p["det_size"], p["det_threshold"])
        self.tracker = FaceTracker(max_misses=p["max_misses"])
        self.emotion = EmotionEstimator(budget_per_frame=p["emotion_budget"])

        self.hostname = socket.gethostname().split(".")[0].upper()
        self.on_source_change = None  # optional callback(spec) after a successful swap

        # The output matrix: four content types x two transports. Names
        # are fixed at launch; whether each feed runs is a live param
        # (_sync_outputs creates/destroys them mid-run).
        base = args.ndi_name
        self.ndi_name = base
        self.ndi_feed_names = {
            "program": base,
            "overlay": args.ndi_overlay or f"{base} Overlay",
            "faces": f"{base} Faces",
            "mask": f"{base} Mask",
        }
        # Syphon/Spout server names follow --ndi-name so a second instance
        # (which must use its own --ndi-name) can't collide; the default
        # stays plain "yewee" to keep existing VJ patches working.
        tex_base = "yewee" if base == "Yewee" else f"yewee-{base}"
        self.tex_feed_names = {
            "program": tex_base,
            "overlay": f"{tex_base}-overlay",
            "faces": f"{tex_base}-faces",
            "mask": f"{tex_base}-mask",
        }
        from . import texture_out
        self.texture_kind, self.texture_error = texture_out.probe()
        self.ndi_outs: dict = {}
        self.tex_outs: dict = {}

        self.source_spec = args.source
        self.startup_error = ""
        try:
            w, h, fps = self._cap_settings(p)
            self.source = open_source(args.source, w, h, fps,
                                      p["cap_backend"], loop=p["loop_file"])
        except Exception as exc:
            self.startup_error = f"source '{args.source}': {exc}"
            self.source = NullSource(args.width or 1280, args.height or 720)

        self._connections: dict = {}
        self._conn_check_frame = -999
        self._perf: dict = {}  # per-stage ms EMAs for the panel meter

        self._stop = threading.Event()
        self.paused = False
        self.restart_requested = False
        self._signal_lost_at: float | None = None
        self._reopen_at = 0.0
        self._slate_cache: dict = {}
        self._card_cache: tuple | None = None
        self._last_size: tuple[int, int] = (args.width or 1280, args.height or 720)
        self._stats_lock = threading.Lock()
        self._stats: dict = {}
        self._pending_source: str | None = None
        self._source_lock = threading.Lock()
        self.last_error = self.startup_error
        self._error_time = time.monotonic() + 3600 if self.startup_error else 0.0

        self._pv_cond = threading.Condition()
        self._pv_seq = 0
        self._pv_jpeg: bytes | None = None
        self._pv_time = 0.0
        self._checker: np.ndarray | None = None  # alpha-preview backdrop
        self.preview_clients = 0        # MJPEG viewers (maintained by webui)
        self.heartbeat = time.monotonic()  # watchdog liveness signal
        self._t0 = time.time()
        self._color_cache: tuple[str, tuple | None] = ("", None)
        self._licence: dict = {"state": "unrestricted"}
        self._licence_checked = 0.0
        self._out_fps_applied = p["out_fps"]   # feeds declare this at creation
        self._cpu_limited: bool | None = None  # last applied limit_cpu
        self._relief = 0                # auto-relief step, 0 = full quality
        self._over_since: float | None = None
        self._under_since: float | None = None
        self._people_models: dict = {}  # lazy, keyed by people_model param
        self._people_soft = False       # whether the active mask is a true matte
        self._people_cache: np.ndarray | None = None
        self._people_roi: tuple[float, float, float, float] | None = None

    # ---- control surface (called from web threads) ----

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def request_restart(self) -> None:
        self.restart_requested = True
        self._stop.set()

    def request_source(self, spec: str) -> None:
        """Applying the SAME spec again is a deliberate reconnect (fresh
        open with the current capture format/backend) — what you want
        after changing those, or when a capture card needs a kick."""
        spec = spec.strip()
        if spec:
            with self._source_lock:
                self._pending_source = spec

    def _cap_settings(self, p: dict) -> tuple[int, int, float]:
        """Requested capture size/rate from the live cap_format param;
        (0, 0, 0) = auto (let the device decide)."""
        return parse_cap_format(p["cap_format"])

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    def wait_preview(self, last_seq: int, timeout: float = 1.0):
        """Blocks until a preview newer than last_seq exists. -> (seq, jpeg) | None"""
        with self._pv_cond:
            if not self._pv_cond.wait_for(lambda: self._pv_seq > last_seq or self.stopped,
                                          timeout=timeout):
                return None
            if self._pv_jpeg is None:
                return None
            return self._pv_seq, self._pv_jpeg

    # ---- internals ----

    CONTENTS = ("program", "overlay", "faces", "mask")

    def _sync_outputs(self, p: dict) -> None:
        """Create/destroy feeds to match the live params — same logic for
        both transports. Cheap no-op when nothing changed. NDI is imported
        only when actually turning a feed on, so NDI-less environments
        (CI) can still run the pipeline with feeds off."""
        if p["out_fps"] != self._out_fps_applied:
            # NDI senders declare their rate at creation; drop them so the
            # loop below rebuilds at the new one (receivers reconnect).
            self._out_fps_applied = p["out_fps"]
            for c in list(self.ndi_outs):
                try:
                    self.ndi_outs.pop(c).close()
                except Exception:
                    pass
        for c in self.CONTENTS:
            try:
                if p[f"ndi_{c}"] and c not in self.ndi_outs:
                    from .ndi_io import NDIOutput
                    self.ndi_outs[c] = NDIOutput(self.ndi_feed_names[c],
                                                 fps=p["out_fps"])
                elif not p[f"ndi_{c}"] and c in self.ndi_outs:
                    self.ndi_outs.pop(c).close()
            except Exception as exc:
                self.last_error = f"NDI output: {exc}"
                self._error_time = time.monotonic()
            want_tex = p[f"tex_{c}"] and bool(self.texture_kind)
            if want_tex and c not in self.tex_outs:
                try:
                    from . import texture_out
                    self.tex_outs[c] = texture_out.create(self.tex_feed_names[c])
                except Exception as exc:
                    self.texture_error = str(exc)
                    self.texture_kind = ""  # don't retry every frame
                    self.last_error = f"Texture share: {exc}"
                    self._error_time = time.monotonic()
            elif not want_tex and c in self.tex_outs:
                self.tex_outs.pop(c).close()

    def _receiver_counts(self, frame_idx: int) -> dict:
        """Connected-receiver counts per NDI feed, refreshed ~3x/second."""
        if frame_idx - self._conn_check_frame >= 10:
            self._conn_check_frame = frame_idx
            counts = {}
            for c, out in self.ndi_outs.items():
                try:
                    counts[c] = int(out.sender.get_num_connections(0)) \
                        if out.sender is not None else 0
                except Exception:
                    counts[c] = 0
            self._connections = counts
        return self._connections

    def _send_idle(self, p: dict, program_img) -> None:
        """Push an idle state to every active feed: `program_img` on the
        program feeds, transparency on the alpha feeds, and an empty mask
        on the mask feeds (black or transparent per style)."""
        _, transparent = self._standby_frames("")
        black, _ = self._standby_frames("")
        mask_img = black if p["mask_style"] == "white" else transparent
        table = {"program": program_img, "overlay": transparent,
                 "faces": transparent, "mask": mask_img}
        for outs in (self.ndi_outs, self.tex_outs):
            for c, out in list(outs.items()):
                try:
                    out.send(table[c])
                except Exception:
                    pass

    def _standby_frames(self, title: str = "STANDBY",
                        sub: str = "resume from the control panel"):
        """(slate BGR, transparent BGRA) at the last known frame size.
        An empty title gives plain black — what live outputs should show
        on a fault (messages are for the panel, not the LED wall)."""
        w, h = self._last_size
        key = (w, h, title)
        if key not in self._slate_cache:
            if title:
                slate = np.full((h, w, 3), (28, 24, 20), dtype=np.uint8)
                for i, (line, scale) in enumerate([(title, 1.6), (sub, 0.7)]):
                    (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                    cv2.putText(slate, line, ((w - tw) // 2, h // 2 + i * 54),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, (150, 150, 160), 2,
                                cv2.LINE_AA)
            else:
                slate = np.zeros((h, w, 3), dtype=np.uint8)
            transparent = np.zeros((h, w, 4), dtype=np.uint8)
            # keep only slates for the current frame size
            self._slate_cache = {k: v for k, v in self._slate_cache.items()
                                 if k[:2] == (w, h)}
            self._slate_cache[key] = (slate, transparent)
        return self._slate_cache[key]

    def _run_paused_tick(self, p: dict) -> None:
        """One loop iteration while paused: keep feeds up with a standby
        slate (overlay goes fully transparent), keep the panel informed."""
        self._sync_outputs(p)
        slate, _ = self._standby_frames()
        self._send_idle(p, slate)
        if self.web_enabled and p["panel_preview"]:
            self._publish_preview(slate)
        with self._stats_lock:
            self._stats.update({"state": "paused", "fps": 0.0, "faces": 0,
                                "error": self.last_error})
        time.sleep(0.1)

    def _run_test_card_tick(self, p: dict) -> None:
        """Send the test pattern to every active feed, with motion (a
        sweeping block and a wall clock) so a frozen link is obvious."""
        self._sync_outputs(p)
        w, h = self._last_size
        if self._card_cache is None or self._card_cache[0] != (w, h):
            base, ovl = render_test_card(w, h, [
                "yewee TEST CARD",
                f"{self.hostname} · {self.ndi_name}",
                f"{w}x{h} @ {p['out_fps']:g} fps target",
            ])
            self._card_cache = ((w, h), base, ovl)
        card = self._card_cache[1].copy()
        ovl = self._card_cache[2].copy()
        t = time.monotonic()
        bw = max(40, w // 16)
        bx = int((t * w / 5.0) % max(1, w - bw))
        by = int(h * 0.73)
        cv2.rectangle(card, (bx, by), (bx + bw, by + max(10, h // 36)),
                      (255, 255, 255), -1)
        cv2.rectangle(ovl, (bx, by), (bx + bw, by + max(10, h // 36)),
                      (255, 255, 255, 255), -1)
        clock = time.strftime("%H:%M:%S")
        fscale = max(0.5, w / 1600)
        th = max(1, round(w / 1000))
        (tw, _), _ = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, fscale * 1.5, th)
        cv2.putText(card, clock, (w - tw - int(w * 0.03), int(h * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale * 1.5, (235, 235, 235), th,
                    cv2.LINE_AA)
        mask_test = (cv2.cvtColor(ovl[:, :, 3], cv2.COLOR_GRAY2BGR)
                     if p["mask_style"] == "white" else ovl)
        table = {"program": card, "overlay": ovl, "faces": ovl, "mask": mask_test}
        for outs in (self.ndi_outs, self.tex_outs):
            for c, out in list(outs.items()):
                try:
                    out.send(table[c])
                except Exception:
                    pass
        if self.web_enabled and p["panel_preview"]:
            self._publish_preview(card)
        with self._stats_lock:
            self._stats.update({"state": "test-card", "fps": 0.0, "faces": 0,
                                "error": self.last_error})
        time.sleep(1 / 30)

    def licence(self) -> dict:
        """Cached licence status — re-read occasionally so activating in
        the panel takes effect without a restart."""
        now = time.monotonic()
        if now - self._licence_checked > 20.0:
            self._licence_checked = now
            try:
                from .licensing import status
                self._licence = status()
            except Exception:            # never let licensing break the show
                self._licence = {"state": "unrestricted"}
        return self._licence

    def refresh_licence(self) -> dict:
        """Force a re-read (called right after the panel activates a key)."""
        self._licence_checked = 0.0
        return self.licence()

    def _run_unlicensed_tick(self, p: dict) -> None:
        """Trial is over: hold the feeds up with a slate that says so,
        rather than dropping them, so the operator can see why."""
        self._sync_outputs(p)
        slate, _ = self._standby_frames(
            "TRIAL ENDED", "enter a licence key in the control panel")
        self._send_idle(p, slate)
        if self.web_enabled and p["panel_preview"]:
            self._publish_preview(slate)
        with self._stats_lock:
            self._stats.update({"state": "unlicensed", "fps": 0.0, "faces": 0,
                                "licence": self._licence, "error": self.last_error})
        time.sleep(0.2)

    def _run_signal_lost_tick(self) -> None:
        """Input died mid-run (unplugged camera, dead NDI feed): keep the
        feeds up with a NO SIGNAL slate, tell the panel, and try to reopen
        the source every few seconds until it comes back."""
        now = time.monotonic()
        p = self.params.snapshot()
        self._sync_outputs(p)
        # Outputs get plain black — graceful on a live screen. The panel
        # preview keeps the diagnostic slate for the operator.
        black, _ = self._standby_frames("")
        self._send_idle(p, black)
        if self.web_enabled and p["panel_preview"]:
            slate, _ = self._standby_frames(
                "NO SIGNAL", f"input '{self.source_spec}' lost - reconnecting")
            self._publish_preview(slate)
        self.last_error = f"Signal lost on '{self.source_spec}' — reconnecting…"
        self._error_time = now  # keep the message alive until recovery
        with self._stats_lock:
            self._stats.update({"state": "no-signal", "fps": 0.0, "faces": 0,
                                "error": self.last_error})
        if now >= self._reopen_at:
            self._reopen_at = now + 3.0
            w, h, fps = self._cap_settings(p)
            try:
                fresh = open_source(self.source_spec, w, h, fps,
                                    p["cap_backend"], loop=p["loop_file"])
            except Exception:
                return  # still gone; keep the slate up
            try:
                self.source.close()
            except Exception:
                pass
            self.source = fresh

    # Relief steps: 1 = segment less often, 2 = also detect every other
    # frame, 3 = also cap detector input. Applied as an internal override
    # so the operator's own settings are never rewritten.
    MAX_RELIEF = 3

    def _sync_detector(self, p: dict) -> None:
        """Swap the detection engine when the panel asks. A backend that
        won't load (no GPU runtime, missing model) falls back to YuNet
        with a panel error rather than taking the show down."""
        if p["detector"] == self._detector_choice:
            return
        want = p["detector"]
        try:
            self.detector = pick_backend(want, p["det_size"], p["det_threshold"])
            self._detector_choice = want
        except Exception as exc:
            self._detector_choice = "yunet"
            self.params.set("detector", "yunet")
            self.last_error = f"detector '{want}' unavailable: {exc} — using YuNet"
            self._error_time = time.monotonic()
            try:
                self.detector = pick_backend("yunet", p["det_size"], p["det_threshold"])
            except Exception:
                pass

    def _apply_cpu_limit(self, p: dict) -> None:
        """Track the limit_cpu param; reload models on change so their
        ONNX thread pools pick up the new budget."""
        if p["limit_cpu"] == self._cpu_limited:
            return
        from .runtime import limit_threads
        limit_threads(p["limit_cpu"])
        self._cpu_limited = p["limit_cpu"]
        self._people_models.clear()

    def _update_relief(self, p: dict, load_pct: int, now: float) -> None:
        """Shed quality when the machine can't hold the frame budget, and
        give it back once there's headroom again. Hysteresis both ways so
        it settles instead of oscillating."""
        if not p["auto_relief"]:
            self._relief = 0
            self._over_since = self._under_since = None
            return
        if load_pct > 100:
            self._under_since = None
            if self._over_since is None:
                self._over_since = now
            elif now - self._over_since > 5.0 and self._relief < self.MAX_RELIEF:
                self._relief += 1
                self._over_since = now
                self.last_error = (f"Auto relief step {self._relief}: reduced "
                                   "quality to keep up with the frame rate")
                self._error_time = now
        elif load_pct < 70:
            self._over_since = None
            if self._under_since is None:
                self._under_since = now
            elif now - self._under_since > 20.0 and self._relief > 0:
                self._relief -= 1
                self._under_since = now
        else:
            self._over_since = self._under_since = None

    def _relieved(self, p: dict) -> dict:
        """The live params with any auto-relief overrides applied."""
        if not self._relief:
            return p
        p = dict(p)
        if self._relief >= 2:
            p["detect_every"] = max(p["detect_every"], 2)
        if self._relief >= 3:
            p["det_size"] = min(p["det_size"], 640)
        return p

    def _people_roi_for(self, frame, tracks):
        """Body-shaped region around the tracked faces: the portrait
        segmenter works far better when people fill its input. Faces are
        heads, so expand sideways and well downward; smooth the box over
        time so the crop doesn't breathe. None = no faces, full frame."""
        H, W = frame.shape[:2]
        if not tracks:
            self._people_roi = None
            return None
        # generous body proportions: exclude the irrelevant, don't hug —
        # a cropped-off limb shows as a hard mask edge
        x1 = min(t.bbox[0] - t.bbox[2] * 3.5 for t in tracks)
        x2 = max(t.bbox[0] + t.bbox[2] * 4.5 for t in tracks)
        y1 = min(t.bbox[1] - t.bbox[3] * 2.0 for t in tracks)
        y2 = max(t.bbox[1] + t.bbox[3] * 8.0 for t in tracks)
        # keep context: never crop tighter than 60% of each dimension
        if x2 - x1 < W * 0.6:
            cx = (x1 + x2) / 2
            x1, x2 = cx - W * 0.3, cx + W * 0.3
        if y2 - y1 < H * 0.6:
            cy = (y1 + y2) / 2
            y1, y2 = cy - H * 0.3, cy + H * 0.3
        box = (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))
        if self._people_roi is not None:
            a = 0.8  # ROI easing
            box = tuple(a * o + (1 - a) * n for o, n in zip(self._people_roi, box))
        self._people_roi = box
        return tuple(int(round(v)) for v in box)

    def _people_mask(self, frame, tracks, steady: float, model_name: str):
        """Mask from the selected silhouette model, loading it on first
        use. Temporal smoothing happens here in full-frame space (correct
        even while the ROI follows the subject). Failures fall back:
        modnet/rvm -> pphumanseg -> oval shape, always with a panel error
        — picking a broken model can't take the feed down."""
        model = self._people_models.get(model_name)
        if model is None:
            try:
                from .segmenter import create_people_model
                model = create_people_model(model_name)
                self._people_models[model_name] = model
            except Exception as exc:
                if model_name != "pphumanseg":
                    self.params.set("people_model", "pphumanseg")
                    self.last_error = f"'{model_name}' unavailable: {exc} — using Fast"
                else:
                    self.params.set("cutout_shape", "oval")
                    self.last_error = f"People cutout unavailable: {exc}"
                self._error_time = time.monotonic()
                return None
        try:
            mask = model.mask(frame, roi=self._people_roi_for(frame, tracks))
            self._people_soft = model.soft
        except Exception as exc:
            self._people_models.pop(model_name, None)
            if model_name != "pphumanseg":
                self.params.set("people_model", "pphumanseg")
                self.last_error = f"'{model_name}' failed: {exc} — using Fast"
            else:
                self.params.set("cutout_shape", "oval")
                self.last_error = f"People cutout failed: {exc}"
            self._error_time = time.monotonic()
            return None
        prev = self._people_cache
        if steady > 0 and prev is not None and prev.shape == mask.shape:
            mask = cv2.addWeighted(prev, steady, mask, 1 - steady, 0)
        return mask

    def _maybe_swap_source(self) -> None:
        with self._source_lock:
            spec, self._pending_source = self._pending_source, None
        if not spec:
            return
        p = self.params.snapshot()
        w, h, fps = self._cap_settings(p)
        try:
            new_source = open_source(spec, w, h, fps, p["cap_backend"],
                                     loop=p["loop_file"])
        except Exception as exc:
            self.last_error = f"source '{spec}': {exc}"
            self._error_time = time.monotonic()
            return
        old = self.source
        self.source = new_source
        self.source_spec = spec
        self.tracker = FaceTracker(max_misses=self.tracker.max_misses)
        self.last_error = ""
        try:
            old.close()
        except Exception:
            pass
        if self.on_source_change is not None:
            try:
                self.on_source_change(spec)
            except Exception:
                pass

    def _brand_color(self, spec: str):
        """'#rrggbb' -> BGR tuple, else None (per-ID palette). Cached."""
        if self._color_cache[0] != spec:
            color = None
            s = spec.strip().lstrip("#")
            if len(s) == 6:
                try:
                    color = (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))
                except ValueError:
                    color = None
            self._color_cache = (spec, color)
        return self._color_cache[1]

    def _publish_preview(self, display) -> None:
        """Publish a frame to the panel. BGRA input is composited over a
        checkerboard so transparency is visible in the (opaque) JPEG.
        Skipped entirely when no browser is streaming the preview."""
        if self.preview_clients < 1:
            return
        now = time.monotonic()
        if now - self._pv_time < PREVIEW_INTERVAL:
            return
        self._pv_time = now
        img = display
        if img.shape[1] > PREVIEW_WIDTH:
            h = int(round(img.shape[0] * PREVIEW_WIDTH / img.shape[1]))
            img = cv2.resize(img, (PREVIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
        if img.shape[2] == 4:
            if self._checker is None or self._checker.shape[:2] != img.shape[:2]:
                h2, w2 = img.shape[:2]
                yy, xx = np.mgrid[0:h2, 0:w2]
                self._checker = np.where(((yy // 24 + xx // 24) % 2)[..., None],
                                         66, 46).astype(np.uint8).repeat(3, axis=2)
            # content is premultiplied: composite = fg + bg * (1 - a)
            a = img[:, :, 3:4].astype(np.uint16)
            img = (img[:, :, :3].astype(np.uint16)
                   + self._checker.astype(np.uint16) * (255 - a) // 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._pv_cond:
                self._pv_seq += 1
                self._pv_jpeg = buf.tobytes()
                self._pv_cond.notify_all()

    def run(self) -> int:
        """Blocking loop; call from the main thread. Returns frame count."""
        args = self.args
        frame_idx = 0
        fps_ema = 0.0
        dt_ema = None
        proc_ema = 0.0
        t_last = time.perf_counter()
        window_open = False
        pace_next = None  # file playback pacing (real-time unless benchmarking)

        try:
            while not self._stop.is_set():
                self.heartbeat = time.monotonic()
                if self.paused:
                    self._run_paused_tick(self.params.snapshot())
                    t_last = time.perf_counter()  # don't count the pause in fps
                    continue
                if self.licence()["state"] == "expired":
                    self._run_unlicensed_tick(self.params.snapshot())
                    t_last = time.perf_counter()
                    continue
                self._maybe_swap_source()
                p = self.params.snapshot()
                if p["test_card"]:
                    self._run_test_card_tick(p)
                    t_last = time.perf_counter()  # don't count card time in fps
                    continue
                in_loss = self._signal_lost_at is not None
                ok, frame = self.source.read(timeout=0.25 if in_loss else 2.0)
                if not ok:
                    if not self.source.is_live:
                        break  # file finished
                    now = time.monotonic()
                    if self._signal_lost_at is None:
                        self._signal_lost_at = now       # brief glitch: just retry
                        self._reopen_at = now + 3.0
                    elif now - self._signal_lost_at > 2.0:
                        self._run_signal_lost_tick()     # it's really gone
                    continue
                if in_loss:
                    self._signal_lost_at = None
                    if self.last_error.startswith("Signal lost"):
                        self.last_error = ""
                    t_last = time.perf_counter()  # don't count the outage in fps
                self._apply_cpu_limit(p)
                self._sync_detector(p)
                p = self._relieved(p)
                self._sync_outputs(p)
                self._last_size = (frame.shape[1], frame.shape[0])
                if p["flip"]:
                    frame = cv2.flip(frame, 1)

                laps: dict = {}
                t0 = time.perf_counter()
                self.detector.apply_live(p["det_threshold"], p["det_size"])
                self.tracker.max_misses = p["max_misses"]
                dets = None
                if frame_idx % p["detect_every"] == 0:
                    dets = self.detector.detect(frame)
                    if p["min_face"] > 0 and len(dets):
                        dets = dets[(dets[:, 2] >= p["min_face"]) & (dets[:, 3] >= p["min_face"])]
                laps["detect"] = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                tracks = self.tracker.step(dets)
                laps["track"] = (time.perf_counter() - t0) * 1000.0
                if p["emotion_enabled"] and p["emotion_budget"] > 0:
                    t0 = time.perf_counter()
                    try:
                        self.emotion.budget = p["emotion_budget"]
                        self.emotion.update(frame, tracks, frame_idx)
                    except Exception as exc:
                        self.params.set("emotion_enabled", False)
                        self.last_error = f"Expressions disabled: {exc}"
                        self._error_time = time.monotonic()
                    laps["express"] = (time.perf_counter() - t0) * 1000.0
                proc_ms = sum(laps.values())
                proc_ema = proc_ms if frame_idx == 0 else 0.9 * proc_ema + 0.1 * proc_ms

                pv_on = self.web_enabled and p["panel_preview"]
                pv_src = p["preview_source"]

                def wants(c):
                    return p[f"ndi_{c}"] or (p[f"tex_{c}"] and bool(self.texture_kind))

                overlay_bgra = None
                if wants("overlay") or (pv_on and pv_src == "overlay"):
                    t0 = time.perf_counter()
                    overlay_bgra = render_overlay_bgra(
                        frame.shape[:2], tracks,
                        show_emotion=p["emotion_enabled"], show_ids=p["show_ids"],
                        color=self._brand_color(p["overlay_color"]))
                    laps["overlay"] = (time.perf_counter() - t0) * 1000.0
                faces_bgra = None
                mask_img = None
                need_faces = wants("faces") or (pv_on and pv_src == "faces")
                need_mask = wants("mask") or (pv_on and pv_src == "mask")
                if need_faces or need_mask:
                    people = None
                    if p["cutout_shape"] == "people":
                        # segmentation is the expensive part; every 2nd
                        # frame is indistinguishable and halves the cost
                        # (every 3rd once auto-relief has stepped in)
                        seg_every = 3 if self._relief else 2
                        stale = (self._people_cache is None
                                 or self._people_cache.shape != frame.shape[:2])
                        if stale or frame_idx % seg_every == 0:
                            t0 = time.perf_counter()
                            fresh = self._people_mask(frame, tracks,
                                                      p["cutout_steady"],
                                                      p["people_model"])
                            laps["silhouette"] = (time.perf_counter() - t0) * 1000.0
                            if fresh is not None:
                                self._people_cache = fresh
                        people = self._people_cache
                    t0 = time.perf_counter()
                    alpha = cutout_alpha(frame.shape[:2], tracks,
                                         margin=p["cutout_margin"],
                                         shape=p["cutout_shape"],
                                         feather=p["cutout_feather"],
                                         people_mask=people,
                                         people_soft=self._people_soft,
                                         grow=p["cutout_grow"])
                    if need_faces:
                        hard = (hard_rect_regions(frame.shape[:2], tracks,
                                                  p["cutout_margin"])
                                if p["cutout_shape"] == "rectangle"
                                and p["cutout_feather"] == 0 else None)
                        faces_bgra = apply_cutout(frame, alpha, hard_regions=hard)
                    if need_mask:
                        mask_img = render_mask(alpha, p["mask_style"])
                    laps["cutout"] = (time.perf_counter() - t0) * 1000.0

                now = time.perf_counter()
                dt = now - t_last
                t_last = now
                # Average the frame TIME, not 1/dt: with work that lands on
                # alternate frames (segmentation), averaging instantaneous
                # rates over-weights the cheap frames and reports an fps
                # well above the real one.
                dt_ema = dt if dt_ema is None else 0.9 * dt_ema + 0.1 * dt
                fps_ema = 1.0 / dt_ema if dt_ema > 0 else 0.0

                # Skip annotation entirely when nothing consumes it (previews
                # off + clean main feed): detection -> tracking -> outputs only.
                # The annotated display draws in place on `frame` when the
                # main feed carries graphics — keep a pristine copy if the
                # panel is watching the clean view.
                clean_frame = frame
                if pv_on and pv_src == "clean" and not p["clean_main"]:
                    clean_frame = frame.copy()
                brand = self._brand_color(p["overlay_color"])
                display = None
                if (p["local_preview"] or (pv_on and pv_src == "annotated")
                        or not p["clean_main"]):
                    t0 = time.perf_counter()
                    display = frame.copy() if p["clean_main"] else frame
                    draw_tracks(display, tracks, show_emotion=p["emotion_enabled"],
                                show_ids=p["show_ids"], color=brand)
                    if p["show_stats"]:
                        n_feeds = len(self.ndi_outs) + len(self.tex_outs)
                        draw_stats(display, [
                            f"{fps_ema:5.1f} fps   faces {len(tracks):3d}   proc {proc_ema:5.1f} ms",
                            f"{self.detector.name}   feeds {n_feeds}",
                        ])
                    laps["annotate"] = (time.perf_counter() - t0) * 1000.0

                out_width = p["out_width"]
                scale_cache: dict = {}  # same image scaled once per frame

                def _scaled(img):
                    if not out_width or img.shape[1] == out_width:
                        return img
                    got = scale_cache.get(id(img))
                    if got is None:
                        oh = int(round(img.shape[0] * out_width / img.shape[1]))
                        got = cv2.resize(img, (out_width, oh),
                                         interpolation=cv2.INTER_AREA)
                        scale_cache[id(img)] = got
                    return got

                t0 = time.perf_counter()
                program = frame if p["clean_main"] else display
                content_img = {"program": program, "overlay": overlay_bgra,
                               "faces": faces_bgra, "mask": mask_img}

                def _send_from(outs, c, kind):
                    """Send, and on failure tear the feed down with a panel
                    error — _sync_outputs recreates it next frame, so a
                    transient NDI/texture hiccup can't kill the show."""
                    out = outs.get(c)
                    img = content_img[c]
                    if out is None or img is None:
                        return
                    try:
                        out.send(_scaled(img))
                    except Exception as exc:
                        self.last_error = f"{kind} {c} output error: {exc} — restarting feed"
                        self._error_time = time.monotonic()
                        try:
                            out.close()
                        except Exception:
                            pass
                        outs.pop(c, None)

                for c in self.CONTENTS:
                    _send_from(self.ndi_outs, c, "NDI")
                    _send_from(self.tex_outs, c, self.texture_kind or "texture")
                laps["outputs"] = (time.perf_counter() - t0) * 1000.0

                if pv_on:
                    t0 = time.perf_counter()
                    pv_img = {"annotated": display, "clean": clean_frame,
                              "overlay": overlay_bgra, "faces": faces_bgra,
                              "mask": mask_img}[pv_src]
                    if pv_img is not None:
                        self._publish_preview(pv_img)
                    laps["preview"] = (time.perf_counter() - t0) * 1000.0
                if p["local_preview"] and display is not None:
                    try:
                        cv2.imshow("yewee (q to quit)", _scaled(display))
                        window_open = True
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except cv2.error:
                        self.params.set("local_preview", False)
                        self.last_error = ("Preview window unavailable on this "
                                           "machine (running headless?)")
                        self._error_time = time.monotonic()
                        window_open = False
                elif window_open:
                    cv2.destroyAllWindows()
                    window_open = False

                if self.last_error and time.monotonic() - self._error_time > 20:
                    self.last_error = ""
                conns = self._receiver_counts(frame_idx)
                # per-stage cost EMAs for the panel's performance meter
                for k in set(self._perf) | set(laps):
                    self._perf[k] = (0.9 * self._perf.get(k, laps.get(k, 0.0))
                                     + 0.1 * laps.get(k, 0.0))
                budget_ms = 1000.0 / (p["out_fps"] or 30)
                load_ms = sum(self._perf.values())
                self._update_relief(p, int(round(load_ms / budget_ms * 100)),
                                    time.monotonic())
                ow = out_width or frame.shape[1]
                oh = int(round(frame.shape[0] * ow / frame.shape[1]))
                with self._stats_lock:
                    self._stats = {
                        "state": "live",
                        "fps": round(fps_ema, 1),
                        "faces": len(tracks),
                        "proc_ms": round(proc_ema, 1),
                        "frame": frame_idx,
                        "backend": self.detector.name,
                        "source": self.source_spec,
                        "ndi_feeds": [
                            {"content": c,
                             "name": f"{self.hostname} ({self.ndi_feed_names[c]})",
                             "watching": conns.get(c, 0)}
                            for c in self.CONTENTS if c in self.ndi_outs],
                        "tex_feeds": [
                            {"content": c, "name": self.tex_feed_names[c]}
                            for c in self.CONTENTS if c in self.tex_outs],
                        "out_res": f"{ow}x{oh}",
                        "out_fps_target": p["out_fps"],
                        "texture_kind": self.texture_kind,
                        "no_input": isinstance(self.source, NullSource),
                        "uptime_s": int(time.time() - self._t0),
                        "perf": {k: round(v, 2) for k, v in self._perf.items()
                                 if v >= 0.02},
                        "budget_ms": round(budget_ms, 1),
                        "load_pct": int(round(load_ms / budget_ms * 100)),
                        "relief": self._relief,
                        "cpu_threads": _rt.budget(),
                        "cv_threads": _rt.cv_threads(),
                        "cpu_cores": _rt.cores(),
                        "people_models": _people_model_choices(),
                        "licence": self.licence(),
                        "error": self.last_error,
                    }

                # Frame-rate ceiling: never run faster than the source
                # supplies (a 30 fps camera caps the loop at 30, a 50 fps
                # one at 50). Nothing downstream benefits from re-running
                # the pipeline between frames, and it keeps the machine
                # cool. Unpaced only for --max-frames benchmark runs.
                if not args.max_frames:
                    src_fps = min(max(getattr(self.source, "fps", 0) or 30.0, 1.0), 120.0)
                    period = 1.0 / src_fps
                    now2 = time.perf_counter()
                    if pace_next is None:
                        pace_next = now2
                    pace_next += period
                    if pace_next > now2:
                        time.sleep(pace_next - now2)
                    else:
                        # behind schedule: reset rather than bank the debt,
                        # or the loop bursts to "catch up" and runs hot
                        pace_next = now2
                else:
                    pace_next = None

                frame_idx += 1
                if not args.quiet and frame_idx % 150 == 0:
                    print(f"[yewee] {fps_ema:5.1f} fps | faces {len(tracks):3d} | "
                          f"proc {proc_ema:5.1f} ms | frame {frame_idx}")
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            self._stop.set()
            with self._pv_cond:
                self._pv_cond.notify_all()
            self.source.close()
            for outs in (self.ndi_outs, self.tex_outs):
                for out in outs.values():
                    try:
                        out.close()
                    except Exception:
                        pass
                outs.clear()
            if window_open:
                cv2.destroyAllWindows()

        if dt_ema is not None:
            print(f"[yewee] done: {frame_idx} frames, {fps_ema:.1f} fps avg (ema), "
                  f"proc {proc_ema:.1f} ms")
        return frame_idx
