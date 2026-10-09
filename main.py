#!/usr/bin/env python3
"""Real-time face detection + tracking with NDI output and a web control panel.

Settings changed in the control panel are saved automatically and restored
on the next launch; CLI flags override them for a single run.

Examples:
  python main.py                              # camera -> NDI, panel on :8089
  python main.py --doctor                     # run the self-check
  python main.py --source "ndi:PTZ Cam 1"    # take an NDI feed as input
  python main.py --clean-main --ndi-overlay "Yewee Overlay"
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import webbrowser

from yewee import settings
from yewee.params import LiveParams
from yewee.pipeline import Pipeline

# The detection defaults are the panel's "Mid crowd" preset (PRESETS.mid in
# yewee/static/index.html), which the panel labels the good default, so a
# fresh install lights that preset. Change the two together.
DEFAULTS = dict(detector="auto", out_fps=30.0, loop_file=True,
                det_threshold=0.45, det_size=640, detect_every=1, min_face=0,
                max_misses=15, emotion_enabled=False, emotion_budget=4,
                show_ids=True, show_stats=True, overlay_color="",
                clean_main=False, flip=False,
                cap_format="1280x720@30", cap_backend="any",
                ndi_program=True, ndi_overlay=False, ndi_faces=False,
                ndi_mask=False, tex_program=False, tex_overlay=False,
                tex_faces=False, tex_mask=False, mask_style="white",
                flip_ndi=False, flip_tex=False,
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
    out.add_argument("--ndi-name", default="Yewee", help="NDI source name")
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
    web.add_argument("--pin", default=None,
                     help="the PIN other devices must give the control panel, kept "
                          "for next time; 'none' turns it off (also kept). Default: "
                          "the saved PIN, or four random digits made on first run "
                          "(\"pin\" in settings.json). The machine itself never "
                          "needs it")

    p.add_argument("--doctor", action="store_true", help="run the self-check and exit")
    # set by the watchdog on Windows when it relaunches a stalled app
    p.add_argument("--wait-for-pid", type=int, default=0, help=argparse.SUPPRESS)
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
        # Off unless the operator switches it on: the product page promises
        # "expression labels, if switched on", and inferring emotions from
        # faces is a regulated use in the EU (see the release review).
        emotion_enabled=False if args.no_emotion
                        else saved_params.get("emotion_enabled", DEFAULTS["emotion_enabled"]),
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
        ndi_mask=False if args.no_ndi else saved_params.get("ndi_mask", False),
        tex_program=True if args.texture_share
                    else saved_params.get("tex_program", False),
        tex_overlay=saved_params.get("tex_overlay", False),
        tex_faces=saved_params.get("tex_faces", False),
        tex_mask=saved_params.get("tex_mask", False),
        mask_style=saved_params.get("mask_style", "white"),
        flip_ndi=saved_params.get("flip_ndi", False),
        flip_tex=saved_params.get("flip_tex", False),
        ndi_overlay=False if args.no_ndi
                    else (True if args.ndi_overlay else saved_params.get("ndi_overlay", False)),
        out_width=rv(args.out_width, "out_width"),
        ndi_faces=False if args.no_ndi else saved_params.get("ndi_faces", False),
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


def _relaunch_argv(extra=()) -> list[str]:
    """The command line that starts yewee again with this run's options.
    From source that is the interpreter plus sys.argv. A packaged app is
    its own executable and sys.argv[0] is that executable again, so there
    it must not be passed twice: argparse would refuse the stray argument
    and the relaunched app would exit before it started."""
    from yewee.paths import is_frozen
    rest = sys.argv[1:] if is_frozen() else sys.argv
    kept, skip = [], False
    for a in rest:               # a relaunch's own --wait-for-pid is spent
        if skip or a == "--wait-for-pid" or a.startswith("--wait-for-pid="):
            skip = a == "--wait-for-pid"
            continue
        kept.append(a)
    return [sys.executable, *kept, *extra]


def _relaunch_env() -> dict:
    """The environment for that relaunch. PyInstaller's bootloader leaves
    variables that make a copy of the same executable think it is a child
    of this one; this one asks it to start as a separate instance."""
    return {**os.environ, "PYINSTALLER_RESET_ENVIRONMENT": "1"}


def _relaunch_after_stall() -> None:
    """Start a fresh yewee in place of this stalled one. macOS: exec into
    it, keeping the pid, so nothing else can grab the panel port between
    the two. Windows has no exec, so the new process is told to wait for
    this one to be gone before its double-launch check, which would
    otherwise find our still-running panel and bow out."""
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(_relaunch_argv(["--wait-for-pid", str(os.getpid())]),
                             env=_relaunch_env())
        else:
            os.execve(sys.executable, _relaunch_argv(), _relaunch_env())
    except Exception as e:
        print(f"[yewee] watchdog: could not relaunch ({e}) — "
              "start yewee again by hand", flush=True)


def _wait_for_exit(pid: int, timeout: float = 20.0) -> None:
    from yewee.crashguard import pid_alive
    deadline = time.monotonic() + timeout
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)


def _start_watchdog(pipeline) -> None:
    """If the pipeline loop wedges (driver stall, blocked I/O) for 30s,
    end this process and get yewee running again. From source the
    launcher's crash loop restarts it on the non-zero exit; the installed
    app has no launcher around it, so it starts its own replacement."""
    from yewee.paths import is_frozen

    def watch():
        while not pipeline.stopped:
            time.sleep(5)
            if (not pipeline.stopped
                    and time.monotonic() - pipeline.heartbeat > 30):
                print("[yewee] watchdog: pipeline stalled for 30s — "
                      "restarting", flush=True)
                from yewee import crashguard
                crashguard.record_hang("The pipeline stopped responding for 30 "
                                       "seconds and the watchdog restarted yewee")
                if is_frozen():
                    _relaunch_after_stall()
                os._exit(3)
    threading.Thread(target=watch, daemon=True, name="yewee-watchdog").start()


def _already_running(port: int) -> bool:
    """True if another yewee instance already serves the panel port."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.8) as r:
            return b"yewee" in r.read(2048)
    except Exception:
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.doctor:
        from yewee.doctor import main as doctor_main
        return doctor_main([])

    if args.wait_for_pid:
        _wait_for_exit(args.wait_for_pid)

    # Double-launch guard: a second instance with the same feed names can
    # crash inside the NDI library, so if yewee already serves the
    # panel port, just show the existing panel instead of starting again.
    # (Deliberate multi-instance setups use --web-port / --ndi-name.)
    if not args.no_web and _already_running(args.web_port):
        url = f"http://localhost:{args.web_port}"
        print(f"yewee is already running on this machine — control panel: {url}")
        if not args.no_browser:
            webbrowser.open(url)   # the running instance knows its own PIN
        return 0

    from yewee.logging_setup import setup as setup_logging
    setup_logging()

    # Follow this run (crash marker, faulthandler, exception hooks) and learn
    # whether the last one ended badly. Its settings come back below exactly
    # as on any launch; the report on it is only offered, never sent unasked.
    from yewee import app_version, crashguard
    previous_crash = crashguard.begin(app_version())

    saved = settings.load()
    params = build_params(args, saved["params"])
    # Apply the CPU budget before any model loads — ONNX Runtime bakes its
    # thread count into each session at creation.
    from yewee.runtime import limit_threads
    limit_threads(params.snapshot()["limit_cpu"])
    if args.source is None:
        args.source = saved["source"] or "0"

    # Crash reports: names of sources and custom feed names never leave the
    # machine (the scrubber replaces them), and the report on a bad last run
    # is queued only if sending is switched on — otherwise the panel asks.
    from yewee import reporting
    reporting.add_secrets(saved["source"], args.source)
    # The panel PIN: on unless the operator turned it off, since anyone on a
    # venue's Wi-Fi could otherwise change or quit the show. Printed once in
    # the banner and shown on the machine's own panel, and scrubbed from
    # reports (including the one about the last run, built just below).
    panel_pin, _ = settings.panel_pin(args.pin)
    reporting.add_pin(panel_pin, saved["pin"])
    for custom in (args.ndi_name, args.ndi_overlay):
        if custom and not custom.startswith("Yewee"):
            reporting.add_secrets(custom)
    crash_report = reporting.handle_previous_session(previous_crash)

    if sys.platform == "darwin":
        # First-ever run: put the macOS camera prompt up now and wait for the
        # answer, on the main thread. Opening a camera before the user has
        # replied just fails, and the prompt needs this thread's run loop.
        # Asked for even when starting on a non-camera source, so that
        # listing and switching to cameras in the panel works later.
        from yewee.capture import (camera_authorization, camera_permission_holder,
                                   request_camera_access)
        if camera_authorization() == "undetermined":
            print("  Waiting for camera permission — click Allow on the"
                  " macOS prompt.", flush=True)
        if request_camera_access() == "denied":
            print("  ! Camera access is denied — System Settings > Privacy &"
                  f" Security > Camera, allow {camera_permission_holder()}."
                  "\n  ! Starting anyway; other source types still work.",
                  flush=True)

    pipeline = Pipeline(args, params, web_enabled=not args.no_web)
    # shop licences check in with letissier.ie daily, off the startup path
    from yewee.licensing import start_check_ins
    start_check_ins()
    def _source_changed(spec: str) -> None:
        settings.save(source=spec)
        reporting.add_secrets(spec)

    pipeline.on_source_change = _source_changed
    # errors yewee recovers from are reported only when sending is on
    pipeline.on_frame_error = reporting.note_nonfatal
    crashguard.on_thread_exception = reporting.note_nonfatal
    reporting.flush_in_background(delay=5.0)    # consented reports, off the startup path

    panel_url = None
    web_server = None
    lan = _lan_ip() if not args.no_web and args.web_host == "0.0.0.0" else None
    phone_url = f"http://{lan}:{args.web_port}" if lan else None
    if not args.no_web:
        from yewee.webui import create_app, start_in_thread
        app = create_app(pipeline, params,
                         on_params_change=settings.save_debounced,
                         pin=panel_pin, phone_url=phone_url)
        web_server = start_in_thread(app, args.web_host, args.web_port)
        panel_url = f"http://localhost:{args.web_port}"

    print(f"\n  yewee {app_version()} is running")
    if panel_url:
        extra = f"   (from other devices: {phone_url})" if phone_url else ""
        print(f"  Control panel : {panel_url}{extra}")
        if panel_pin:
            print(f"  Panel PIN     : {panel_pin}   (other devices ask for it once;"
                  " --pin to change, --pin none to turn off)")
        else:
            print("  Panel PIN     : off — anyone on this network can use the panel"
                  " (--pin to set one)")
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
            from yewee.capture import camera_permission_holder
            print("  ! If this is a permissions issue: System Settings > Privacy & Security"
                  f" > Camera, allow {camera_permission_holder()}, then restart.")
    if previous_crash:
        print(f"\n  ! yewee closed unexpectedly last time ({previous_crash['kind']}): "
              f"{previous_crash['summary'][:160]}"
              "\n  ! Your source, feeds and settings have been restored.")
        if crash_report == "queued":
            print("  ! A crash report will be sent (\"Send crash reports "
                  "automatically\" is on).")
        elif crash_report == "prompt" and panel_url:
            print("  ! The control panel asks whether to send a crash report.")
    print("  Press Ctrl-C to stop.\n", flush=True)

    if panel_url and not args.no_browser:
        # This browser is on the machine that printed the PIN, so hand it
        # over rather than prompting the operator for their own number.
        open_url = f"{panel_url}/?pin={panel_pin}" if panel_pin else panel_url
        t = threading.Timer(1.2, webbrowser.open, args=(open_url,))
        t.daemon = True
        t.start()

    def _stop(signum, frame):
        pipeline.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGHUP"):
        # closing the Terminal window: a deliberate quit, not a crash
        signal.signal(signal.SIGHUP, _stop)
    crashguard.watch_console_close(pipeline.stop)   # the Windows equivalent

    if not args.max_frames:  # not for benchmarks/tests
        _keep_awake()
        _start_watchdog(pipeline)

    pipeline.run()
    crashguard.end_clean()      # everything after this is a deliberate exit

    if pipeline.restart_requested:
        # Relaunch ourselves with the same command line (panel "Restart").
        print("[yewee] restarting...", flush=True)
        if web_server is not None:
            web_server.should_exit = True  # release the panel port first
            time.sleep(0.7)
        argv_full = _relaunch_argv()
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(argv_full, env=_relaunch_env())
            return 0
        os.execve(sys.executable, argv_full, _relaunch_env())
    return 0


if __name__ == "__main__":
    sys.exit(main())
