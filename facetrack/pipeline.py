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

from .capture import NullSource, open_source
from .detectors import pick_backend
from .emotion import EmotionEstimator
from .overlay import (draw_stats, draw_tracks, render_faces_cutout,
                      render_overlay_bgra, render_test_card)
from .params import LiveParams
from .tracker import FaceTracker

PREVIEW_INTERVAL = 0.08   # ~12 fps JPEG preview for the panel
PREVIEW_WIDTH = 640


class Pipeline:
    def __init__(self, args, params: LiveParams, web_enabled: bool = False):
        self.args = args
        self.params = params
        self.web_enabled = web_enabled

        p = params.snapshot()
        self.detector = pick_backend(args.backend, p["det_size"], p["det_threshold"])
        self.tracker = FaceTracker(max_misses=p["max_misses"])
        self.emotion = EmotionEstimator(budget_per_frame=p["emotion_budget"])

        self.hostname = socket.gethostname().split(".")[0].upper()
        self.on_source_change = None  # optional callback(spec) after a successful swap

        # Output names are fixed at launch; whether each output runs is a
        # live param (_sync_outputs creates/destroys them mid-run).
        self.ndi_name = args.ndi_name
        self.overlay_name = args.ndi_overlay or f"{args.ndi_name} Overlay"
        self.faces_name = f"{args.ndi_name} Faces"
        from . import texture_out
        self.texture_kind, self.texture_error = texture_out.probe()
        self.texture = None

        self.source_spec = args.source
        self.startup_error = ""
        try:
            self.source = open_source(args.source, args.width, args.height, args.fps,
                                      args.capture_backend, loop=args.loop)
        except Exception as exc:
            self.startup_error = f"source '{args.source}': {exc}"
            self.source = NullSource(args.width or 1280, args.height or 720)

        self.ndi = None
        self.ndi_overlay = None
        self.ndi_faces = None
        self._connections = (0, 0, 0)
        self._conn_check_frame = -999

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
        spec = spec.strip()
        if spec:
            with self._source_lock:
                self._pending_source = spec

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

    def _sync_outputs(self, p: dict) -> None:
        """Create/destroy NDI senders and the Syphon/Spout server to match
        the live params. Cheap no-op when nothing changed. NDI is imported
        only when actually turning a feed on, so NDI-less environments
        (CI) can still run the pipeline with feeds off."""
        try:
            if p["ndi_main"] and self.ndi is None:
                from .ndi_io import NDIOutput
                self.ndi = NDIOutput(self.ndi_name, fps=self.args.fps)
            elif not p["ndi_main"] and self.ndi is not None:
                self.ndi.close()
                self.ndi = None
            if p["ndi_overlay"] and self.ndi_overlay is None:
                from .ndi_io import NDIOutput
                self.ndi_overlay = NDIOutput(self.overlay_name, fps=self.args.fps)
            elif not p["ndi_overlay"] and self.ndi_overlay is not None:
                self.ndi_overlay.close()
                self.ndi_overlay = None
            if p["ndi_faces"] and self.ndi_faces is None:
                from .ndi_io import NDIOutput
                self.ndi_faces = NDIOutput(self.faces_name, fps=self.args.fps)
            elif not p["ndi_faces"] and self.ndi_faces is not None:
                self.ndi_faces.close()
                self.ndi_faces = None
        except Exception as exc:
            self.last_error = f"NDI output: {exc}"
            self._error_time = time.monotonic()

        want_texture = p["texture_share"] and bool(self.texture_kind)
        if want_texture and self.texture is None:
            try:
                from . import texture_out
                self.texture = texture_out.create("facetrack")
            except Exception as exc:
                self.texture_error = str(exc)
                self.texture_kind = ""  # don't retry every frame
        elif not want_texture and self.texture is not None:
            self.texture.close()
            self.texture = None

    def _receiver_counts(self, frame_idx: int) -> tuple[int, int, int]:
        """Connected-receiver counts per NDI feed, refreshed ~3x/second."""
        if frame_idx - self._conn_check_frame >= 10:
            self._conn_check_frame = frame_idx
            def count(out):
                if out is None or out.sender is None:
                    return 0
                try:
                    return int(out.sender.get_num_connections(0))
                except Exception:
                    return 0
            self._connections = (count(self.ndi), count(self.ndi_overlay),
                                 count(self.ndi_faces))
        return self._connections

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
        slate, transparent = self._standby_frames()
        if self.ndi is not None:
            self.ndi.send(slate)
        if self.ndi_overlay is not None:
            self.ndi_overlay.send(transparent)
        if self.ndi_faces is not None:
            self.ndi_faces.send(transparent)
        if self.texture is not None:
            self.texture.send(slate if p["texture_source"] == "program" else transparent)
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
                "facetrack TEST CARD",
                f"{self.hostname} · {self.ndi_name}",
                f"{w}x{h} @ {self.args.fps:g} fps target",
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
        if self.ndi is not None:
            self.ndi.send(card)
        if self.ndi_overlay is not None:
            self.ndi_overlay.send(ovl)
        if self.ndi_faces is not None:
            self.ndi_faces.send(ovl)
        if self.texture is not None:
            self.texture.send(card if p["texture_source"] == "program" else ovl)
        if self.web_enabled and p["panel_preview"]:
            self._publish_preview(card)
        with self._stats_lock:
            self._stats.update({"state": "test-card", "fps": 0.0, "faces": 0,
                                "error": self.last_error})
        time.sleep(1 / 30)

    def _run_signal_lost_tick(self) -> None:
        """Input died mid-run (unplugged camera, dead NDI feed): keep the
        feeds up with a NO SIGNAL slate, tell the panel, and try to reopen
        the source every few seconds until it comes back."""
        now = time.monotonic()
        p = self.params.snapshot()
        self._sync_outputs(p)
        # Outputs get plain black — graceful on a live screen. The panel
        # preview keeps the diagnostic slate for the operator.
        black, transparent = self._standby_frames("")
        if self.ndi is not None:
            self.ndi.send(black)
        if self.ndi_overlay is not None:
            self.ndi_overlay.send(transparent)
        if self.ndi_faces is not None:
            self.ndi_faces.send(transparent)
        if self.texture is not None:
            self.texture.send(black if p["texture_source"] == "program" else transparent)
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
            a = self.args
            try:
                fresh = open_source(self.source_spec, a.width, a.height, a.fps,
                                    a.capture_backend, loop=True)
            except Exception:
                return  # still gone; keep the slate up
            try:
                self.source.close()
            except Exception:
                pass
            self.source = fresh

    def _maybe_swap_source(self) -> None:
        with self._source_lock:
            spec, self._pending_source = self._pending_source, None
        if not spec or spec == self.source_spec:
            return
        a = self.args
        try:
            new_source = open_source(spec, a.width, a.height, a.fps,
                                     a.capture_backend, loop=True)
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

    def _publish_preview(self, display) -> None:
        now = time.monotonic()
        if now - self._pv_time < PREVIEW_INTERVAL:
            return
        self._pv_time = now
        img = display
        if img.shape[1] > PREVIEW_WIDTH:
            h = int(round(img.shape[0] * PREVIEW_WIDTH / img.shape[1]))
            img = cv2.resize(img, (PREVIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
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
        fps_ema = None
        proc_ema = 0.0
        t_last = time.perf_counter()
        window_open = False
        pace_next = None  # file playback pacing (real-time unless benchmarking)

        try:
            while not self._stop.is_set():
                if self.paused:
                    self._run_paused_tick(self.params.snapshot())
                    t_last = time.perf_counter()  # don't count the pause in fps
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
                self._sync_outputs(p)
                self._last_size = (frame.shape[1], frame.shape[0])
                if p["flip"]:
                    frame = cv2.flip(frame, 1)

                t0 = time.perf_counter()
                self.detector.apply_live(p["det_threshold"], p["det_size"])
                self.tracker.max_misses = p["max_misses"]
                dets = None
                if frame_idx % p["detect_every"] == 0:
                    dets = self.detector.detect(frame)
                    if p["min_face"] > 0 and len(dets):
                        dets = dets[(dets[:, 2] >= p["min_face"]) & (dets[:, 3] >= p["min_face"])]
                tracks = self.tracker.step(dets)
                if p["emotion_enabled"] and p["emotion_budget"] > 0:
                    self.emotion.budget = p["emotion_budget"]
                    self.emotion.update(frame, tracks, frame_idx)
                proc_ms = (time.perf_counter() - t0) * 1000.0
                proc_ema = proc_ms if frame_idx == 0 else 0.9 * proc_ema + 0.1 * proc_ms

                overlay_bgra = None
                need_overlay = (self.ndi_overlay is not None
                                or (self.texture is not None
                                    and p["texture_source"] == "overlay"))
                if need_overlay:
                    overlay_bgra = render_overlay_bgra(
                        frame.shape[:2], tracks,
                        show_emotion=p["emotion_enabled"], show_ids=p["show_ids"])
                faces_bgra = None
                need_faces = (self.ndi_faces is not None
                              or (self.texture is not None
                                  and p["texture_source"] == "faces"))
                if need_faces:
                    faces_bgra = render_faces_cutout(frame, tracks,
                                                     margin=p["cutout_margin"])

                now = time.perf_counter()
                dt = now - t_last
                t_last = now
                inst = 1.0 / dt if dt > 0 else 0.0
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

                # Skip annotation entirely when nothing consumes it (previews
                # off + clean main feed): detection -> tracking -> outputs only.
                display = None
                if (p["local_preview"] or (self.web_enabled and p["panel_preview"])
                        or not p["clean_main"]):
                    display = frame.copy() if p["clean_main"] else frame
                    draw_tracks(display, tracks, show_emotion=p["emotion_enabled"],
                                show_ids=p["show_ids"])
                    if p["show_stats"]:
                        draw_stats(display, [
                            f"{fps_ema:5.1f} fps   faces {len(tracks):3d}   proc {proc_ema:5.1f} ms",
                            f"{self.detector.name}   NDI "
                            f"{'off' if self.ndi is None else self.ndi_name}",
                        ])

                out_width = p["out_width"]

                def _scaled(img):
                    if out_width and img.shape[1] != out_width:
                        oh = int(round(img.shape[0] * out_width / img.shape[1]))
                        return cv2.resize(img, (out_width, oh),
                                          interpolation=cv2.INTER_AREA)
                    return img

                program = frame if p["clean_main"] else display
                if self.ndi is not None:
                    self.ndi.send(_scaled(program))
                if self.ndi_overlay is not None:
                    self.ndi_overlay.send(_scaled(overlay_bgra))
                if self.ndi_faces is not None:
                    self.ndi_faces.send(_scaled(faces_bgra))
                if self.texture is not None:
                    tex_img = {"overlay": overlay_bgra,
                               "faces": faces_bgra}.get(p["texture_source"])
                    if tex_img is None:
                        tex_img = program
                    if tex_img is not None:
                        self.texture.send(_scaled(tex_img))
                if self.web_enabled and p["panel_preview"] and display is not None:
                    self._publish_preview(display)
                if p["local_preview"] and display is not None:
                    try:
                        cv2.imshow("facetrack (q to quit)", _scaled(display))
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
                conn_main, conn_ovl, conn_faces = self._receiver_counts(frame_idx)
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
                        "ndi_name": "" if self.ndi is None else self.ndi_name,
                        "ndi_overlay": "" if self.ndi_overlay is None else self.overlay_name,
                        "ndi_display": "" if self.ndi is None
                                       else f"{self.hostname} ({self.ndi_name})",
                        "ndi_overlay_display": "" if self.ndi_overlay is None
                                               else f"{self.hostname} ({self.overlay_name})",
                        "ndi_faces_display": "" if self.ndi_faces is None
                                             else f"{self.hostname} ({self.faces_name})",
                        "ndi_connections": conn_main,
                        "ndi_overlay_connections": conn_ovl,
                        "ndi_faces_connections": conn_faces,
                        "out_res": f"{ow}x{oh}",
                        "out_fps_target": args.fps,
                        "texture_kind": self.texture_kind,
                        "texture_on": self.texture is not None,
                        "texture_error": self.texture_error
                                         if p["texture_share"] and self.texture is None else "",
                        "no_input": isinstance(self.source, NullSource),
                        "error": self.last_error,
                    }

                # Pace file sources to their native fps so tests behave like
                # a real feed (fans stay quiet, NDI rates stay sane). Runs
                # unpaced when --max-frames is set (benchmark mode).
                if not self.source.is_live and not args.max_frames:
                    fps_native = min(getattr(self.source, "fps", 0) or 30.0, 120.0)
                    period = 1.0 / fps_native
                    now2 = time.perf_counter()
                    pace_next = (pace_next if pace_next is not None else now2) + period
                    if pace_next > now2:
                        time.sleep(pace_next - now2)
                    elif now2 - pace_next > 0.5:
                        pace_next = now2  # fell behind; don't try to catch up
                else:
                    pace_next = None

                frame_idx += 1
                if not args.quiet and frame_idx % 150 == 0:
                    print(f"[facetrack] {fps_ema:5.1f} fps | faces {len(tracks):3d} | "
                          f"proc {proc_ema:5.1f} ms | frame {frame_idx}")
                if args.max_frames and frame_idx >= args.max_frames:
                    break
        finally:
            self._stop.set()
            with self._pv_cond:
                self._pv_cond.notify_all()
            self.source.close()
            if self.ndi is not None:
                self.ndi.close()
            if self.ndi_overlay is not None:
                self.ndi_overlay.close()
            if self.ndi_faces is not None:
                self.ndi_faces.close()
            if self.texture is not None:
                self.texture.close()
            if window_open:
                cv2.destroyAllWindows()

        if fps_ema is not None:
            print(f"[facetrack] done: {frame_idx} frames, {fps_ema:.1f} fps avg (ema), "
                  f"proc {proc_ema:.1f} ms")
        return frame_idx
