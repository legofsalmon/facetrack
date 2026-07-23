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

from .capture import NullSource, open_source
from .detectors import pick_backend
from .emotion import EmotionEstimator
from .overlay import draw_stats, draw_tracks, render_overlay_bgra
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
        self._connections = (0, 0)
        self._conn_check_frame = -999

        self._stop = threading.Event()
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
        the live params. Cheap no-op when nothing changed."""
        from .ndi_io import NDIOutput
        try:
            if p["ndi_main"] and self.ndi is None:
                self.ndi = NDIOutput(self.ndi_name, fps=self.args.fps)
            elif not p["ndi_main"] and self.ndi is not None:
                self.ndi.close()
                self.ndi = None
            if p["ndi_overlay"] and self.ndi_overlay is None:
                self.ndi_overlay = NDIOutput(self.overlay_name, fps=self.args.fps)
            elif not p["ndi_overlay"] and self.ndi_overlay is not None:
                self.ndi_overlay.close()
                self.ndi_overlay = None
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

    def _receiver_counts(self, frame_idx: int) -> tuple[int, int]:
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
            self._connections = (count(self.ndi), count(self.ndi_overlay))
        return self._connections

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

        try:
            while not self._stop.is_set():
                self._maybe_swap_source()
                ok, frame = self.source.read()
                if not ok:
                    if self.source.is_live:
                        continue
                    break
                p = self.params.snapshot()
                self._sync_outputs(p)
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
                                or (self.texture is not None and p["texture_overlay"]))
                if need_overlay:
                    overlay_bgra = render_overlay_bgra(
                        frame.shape[:2], tracks,
                        show_emotion=p["emotion_enabled"], show_ids=p["show_ids"])

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
                if self.texture is not None:
                    tex_img = overlay_bgra if (p["texture_overlay"]
                                               and overlay_bgra is not None) else program
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
                conn_main, conn_ovl = self._receiver_counts(frame_idx)
                ow = out_width or frame.shape[1]
                oh = int(round(frame.shape[0] * ow / frame.shape[1]))
                with self._stats_lock:
                    self._stats = {
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
                        "ndi_connections": conn_main,
                        "ndi_overlay_connections": conn_ovl,
                        "out_res": f"{ow}x{oh}",
                        "out_fps_target": args.fps,
                        "texture_kind": self.texture_kind,
                        "texture_on": self.texture is not None,
                        "texture_error": self.texture_error
                                         if p["texture_share"] and self.texture is None else "",
                        "no_input": isinstance(self.source, NullSource),
                        "error": self.last_error,
                    }

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
            if self.texture is not None:
                self.texture.close()
            if window_open:
                cv2.destroyAllWindows()

        if fps_ema is not None:
            print(f"[facetrack] done: {frame_idx} frames, {fps_ema:.1f} fps avg (ema), "
                  f"proc {proc_ema:.1f} ms")
        return frame_idx
