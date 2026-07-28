#!/usr/bin/env python3
"""Real-time face detection + tracking with NDI output and a web control panel.

Settings changed in the control panel are saved automatically and restored
on the next launch; CLI flags override them for a single run.

Examples:
  python main.py                              # camera -> NDI, panel on :8089
  python main.py --doctor                     # run the self-check
  python main.py --source "ndi:PTZ Cam 1"    # take an NDI feed as input
  python main.py --clean-main --ndi-overlay "FaceTracker Overlay"
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import webbrowser

from facetrack import settings
from facetrack.params import LiveParams
from facetrack.pipeline import Pipeline

DEFAULTS = dict(detector="auto", out_fps=30.0, loop_file=True,
                det_threshold=0.5, det_size=640, detect_every=1, min_face=0,
                max_misses=15, emotion_enabled=True, emotion_budget=4,
                show_ids=True, show_stats=True, overlay_color="",
                clean_main=False, flip=False,
                cap_format="1280x720@30", cap_backend="any",
                ndi_program=True, ndi_overlay=False, ndi_faces=False,
                ndi_mask=False, tex_program=False, tex_overlay=False,
                tex_faces=False, tex_mask=False, mask_style="white",
                out_width=0, cutout_margin=0.15,
                cutout_shape="rectangle", cutout_feather=0, cutout_grow=0,
                cutout_steady=0.55,
                people_model="pphumanseg",
                limit_cpu=False, auto_relief=True,
                test_card=False, panel_preview=True, local_preview=True,
                preview_source="annotated")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--source", default=None,
                     help="camera index, video file/URL, or 'ndi:<source name>' "
                          "(default: last used, else camera 0)")
    src.add_argument("--width", type=int, default=1280, help="requested capture width")
    src.add_argument("--height", type=int, default=720, help="requested capture height")
    src.add_argument("--fps", type=float, default=30.0, help="requested capture fps / NDI output fps")
    src.add_argument("--capture-backend", default="any",
                     choices=["any", "avfoundation", "dshow", "msmf"],
                     help="OpenCV capture backend (dshow/msmf on Windows)")
    src.add_argument("--loop", action="store_true", help="loop video file input")
    src.add_argument("--flip", action="store_const", const=True, default=None,
                     help="mirror the image")

    det = p.add_argument_group("detection / tracking (all live-adjustable in the panel)")
    det.add_argument("--backend", default="auto", choices=["auto", "yunet", "centerface"],
                     help="detector backend (auto: CenterFace on an NVIDIA GPU, else YuNet); also a panel setting")
    det.add_argument("--det-size", type=int, default=None,
                     help="detector input size in px; raise for more small faces")
    det.add_argument("--det-threshold", type=float, default=None,
                     help="detection confidence threshold")
    det.add_argument("--detect-every", type=int, default=None,
                     help="run the detector every N frames")
    det.add_argument("--min-face", type=int, default=None,
                     help="ignore faces smaller than this many px")
    det.add_argument("--max-misses", type=int, default=None,
                     help="frames a track survives without a matching detection")

    emo = p.add_argument_group("emotion")
    emo.add_argument("--no-emotion", action="store_const", const=True, default=None,
                     help="disable expression estimation")
    emo.add_argument("--emotion-budget", type=int, default=None,
                     help="max faces scored for expression per frame")

    out = p.add_argument_group("output")
    out.add_argument("--ndi-name", default="FaceTracker", help="NDI source name")
    out.add_argument("--ndi-overlay", default="", metavar="NAME",
                     help="also send a second NDI source with ONLY the tracking graphics "
                          "on transparency (alpha), for keying downstream")
    out.add_argument("--clean-main", action="store_const", const=True, default=None,
                     help="keep the main NDI feed clean (no graphics burned in)")
    out.add_argument("--no-ndi", action="store_true", help="start with all NDI feeds off")
    out.add_argument("--out-width", type=int, default=None,
                     help="scale output to this width before sending (0 = capture size)")
    out.add_argument("--texture-share", action="store_const", const=True, default=None,
                     help="also publish via Syphon (macOS) / Spout (Windows)")
    out.add_argument("--no-preview", action="store_true",
                     help="start with the local preview window off (also a live panel toggle)")
    out.add_argument("--no-ids", action="store_const", const=True, default=None,
                     help="hide track ID labels")
    out.add_argument("--no-stats", action="store_const", const=True, default=None,
                     help="hide the stats bar")

    web = p.add_argument_group("control panel")
    web.add_argument("--no-web", action="store_true", help="disable the web control panel")
    web.add_argument("--no-browser", action="store_true",
                     help="don't auto-open the panel in a browser")
    web.add_argument("--web-host", default="0.0.0.0",
                     help="control panel bind address (default: all interfaces)")
    web.add_argument("--web-port", type=int, default=8089, help="control panel port")
    web.add_argument("--pin", default="",
                     help="require this PIN in the control panel (also settable as "
                          '"pin" in settings.json)')

    p.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    p.add_argument("--quiet", action="store_true", help="no periodic console stats")
    return p.parse_args(argv)


def _lan_ip() -> str | None:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def build_params(args, saved_params: dict) -> LiveParams:
    def rv(cli, key):
        return cli if cli is not None else saved_params.get(key, DEFAULTS[key])

    return LiveParams(
        # CLI wins over the saved value only when explicitly given
        detector=(args.backend if args.backend != "auto"
                  else saved_params.get("detector", "auto")),
        out_fps=(args.fps if args.fps != 30.0
                 else saved_params.get("out_fps", 30.0)),
        loop_file=True if args.loop else saved_params.get("loop_file", True),
        det_threshold=rv(args.det_threshold, "det_threshold"),
        det_size=rv(args.det_size, "det_size"),
        detect_every=rv(args.detect_every, "detect_every"),
        min_face=rv(args.min_face, "min_face"),
        max_misses=rv(args.max_misses, "max_misses"),
        emotion_enabled=False if args.no_emotion else saved_params.get("emotion_enabled", True),
        emotion_budget=rv(args.emotion_budget, "emotion_budget"),
        show_ids=False if args.no_ids else saved_params.get("show_ids", True),
        show_stats=False if args.no_stats else saved_params.get("show_stats", True),
        clean_main=True if args.clean_main else saved_params.get("clean_main", False),
        flip=True if args.flip else saved_params.get("flip", False),
        # explicit CLI capture flags win over the saved panel values
        cap_format=(f"{args.width}x{args.height}@{args.fps:g}"
                    if (args.width, args.height, args.fps) != (1280, 720, 30.0)
                    else saved_params.get("cap_format", "1280x720@30")),
        cap_backend=(args.capture_backend if args.capture_backend != "any"
                     else saved_params.get("cap_backend", "any")),
        ndi_program=False if args.no_ndi else saved_params.get("ndi_program", True),
        ndi_mask=saved_params.get("ndi_mask", False),
        tex_program=True if args.texture_share
                    else saved_params.get("tex_program", False),
        tex_overlay=saved_params.get("tex_overlay", False),
        tex_faces=saved_params.get("tex_faces", False),
        tex_mask=saved_params.get("tex_mask", False),
        mask_style=saved_params.get("mask_style", "white"),
        ndi_overlay=False if args.no_ndi
                    else (True if args.ndi_overlay else saved_params.get("ndi_overlay", False)),
        out_width=rv(args.out_width, "out_width"),
        ndi_faces=saved_params.get("ndi_faces", False),
        cutout_margin=saved_params.get("cutout_margin", 0.15),
        cutout_shape=saved_params.get("cutout_shape", "rectangle"),
        cutout_feather=saved_params.get("cutout_feather", 0),
        cutout_grow=saved_params.get("cutout_grow", 0),
        cutout_steady=saved_params.get("cutout_steady", 0.55),
        overlay_color=saved_params.get("overlay_color", ""),
        people_model=saved_params.get("people_model", "pphumanseg"),
        limit_cpu=saved_params.get("limit_cpu", False),
        auto_relief=saved_params.get("auto_relief", True),
        test_card=saved_params.get("test_card", False),
        panel_preview=saved_params.get("panel_preview", True),
        preview_source=saved_params.get("preview_source", "annotated"),
        local_preview=False if args.no_preview
                      else saved_params.get("local_preview", True),
    )


def _keep_awake() -> None:
    """Stop the machine sleeping mid-show. macOS: caffeinate tied to our
    pid (dies with us). Windows: SetThreadExecutionState on this thread."""
    try:
        if sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            import ctypes
            es = 0x80000000 | 0x00000001 | 0x00000002  # CONTINUOUS|SYSTEM|DISPLAY
            ctypes.windll.kernel32.SetThreadExecutionState(es)
    except Exception:
        pass  # nice-to-have, never fatal


def _start_watchdog(pipeline) -> None:
    """If the pipeline loop wedges (driver stall, blocked I/O) for 30s,
    exit non-zero so the launcher's crash-restart brings us back."""
    def watch():
        while not pipeline.stopped:
            time.sleep(5)
            if (not pipeline.stopped
                    and time.monotonic() - pipeline.heartbeat > 30):
                print("[facetrack] watchdog: pipeline stalled for 30s — "
                      "exiting so the launcher can restart", flush=True)
                os._exit(3)
    threading.Thread(target=watch, daemon=True, name="facetrack-watchdog").start()


def _already_running(port: int) -> bool:
    """True if another facetrack instance already serves the panel port."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.8) as r:
            return b"facetrack" in r.read(2048)
    except Exception:
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.doctor:
        from facetrack.doctor import main as doctor_main
        return doctor_main([])

    # Double-launch guard: a second instance with the same feed names can
    # crash inside the NDI library, so if facetrack already serves the
    # panel port, just show the existing panel instead of starting again.
    # (Deliberate multi-instance setups use --web-port / --ndi-name.)
    if not args.no_web and _already_running(args.web_port):
        url = f"http://localhost:{args.web_port}"
        print(f"facetrack is already running on this machine — control panel: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    from facetrack.logging_setup import setup as setup_logging
    setup_logging(os.path.dirname(os.path.abspath(__file__)))

    saved = settings.load()
    params = build_params(args, saved["params"])
    # Apply the CPU budget before any model loads — ONNX Runtime bakes its
    # thread count into each session at creation.
    from facetrack.runtime import limit_threads
    limit_threads(params.snapshot()["limit_cpu"])
    if args.source is None:
        args.source = saved["source"] or "0"

    if sys.platform == "darwin":
        # First-ever run: pop the macOS camera prompt right away (attributed
        # to the terminal that launched us) instead of failing silently.
        from facetrack.capture import request_camera_access
        request_camera_access()

    pipeline = Pipeline(args, params, web_enabled=not args.no_web)
    pipeline.on_source_change = lambda spec: settings.save(source=spec)

    panel_url = None
    web_server = None
    panel_pin = args.pin or saved["pin"]
    if not args.no_web:
        from facetrack.webui import create_app, start_in_thread
        app = create_app(pipeline, params,
                         on_params_change=settings.save_debounced,
                         pin=panel_pin)
        web_server = start_in_thread(app, args.web_host, args.web_port)
        panel_url = f"http://localhost:{args.web_port}"

    print("\n  facetrack is running")
    if panel_url:
        lan = _lan_ip()
        extra = f"   (from other devices: http://{lan}:{args.web_port})" \
            if lan and args.web_host == "0.0.0.0" else ""
        print(f"  Control panel : {panel_url}{extra}")
        if panel_pin:
            print("  Panel PIN     : required (set via --pin / settings.json)")
    p0 = params.snapshot()
    notes = {"program": "", "overlay": "  [graphics on alpha]",
             "faces": "  [cutout on alpha]", "mask": "  [matte]"}
    for c in ("program", "overlay", "faces", "mask"):
        if p0[f"ndi_{c}"]:
            print(f"  NDI {c:<9} : {pipeline.hostname} "
                  f"({pipeline.ndi_feed_names[c]}){notes[c]}")
    if pipeline.texture_kind:
        tex_on = [c for c in ("program", "overlay", "faces", "mask") if p0[f"tex_{c}"]]
        state = ", ".join(tex_on) if tex_on else "available (enable in the panel)"
        print(f"  {pipeline.texture_kind.capitalize():<13} : {state}")
    print(f"  Input         : {args.source}   detector: {pipeline.detector.name}")
    if pipeline.startup_error:
        print(f"\n  ! {pipeline.startup_error}")
        print("  ! The app started anyway — pick a working source in the control panel.")
        if sys.platform == "darwin" and args.source.isdigit():
            print("  ! If this is a permissions issue: System Settings > Privacy & Security"
                  " > Camera, allow your terminal app, then restart.")
    print("  Press Ctrl-C to stop.\n", flush=True)

    if panel_url and not args.no_browser:
        t = threading.Timer(1.2, webbrowser.open, args=(panel_url,))
        t.daemon = True
        t.start()

    def _stop(signum, frame):
        pipeline.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if not args.max_frames:  # not for benchmarks/tests
        _keep_awake()
        _start_watchdog(pipeline)

    pipeline.run()

    if pipeline.restart_requested:
        # Relaunch ourselves with the same command line (panel "Restart").
        print("[facetrack] restarting...", flush=True)
        if web_server is not None:
            web_server.should_exit = True  # release the panel port first
            time.sleep(0.7)
        argv_full = [sys.executable] + sys.argv
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(argv_full)
            return 0
        os.execv(sys.executable, argv_full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
