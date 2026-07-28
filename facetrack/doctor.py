"""Self-check for facetrack: `python -m facetrack.doctor [--fix] [--no-camera]`

Verifies Python, packages, models, camera access, detector backend, NDI,
and the control-panel port, in plain language with a fix for anything
broken. --fix downloads any missing model files. Used by the Setup
scripts; safe to run any time.
"""
from __future__ import annotations

import argparse
import socket
import sys
import urllib.request
from pathlib import Path

GREEN, RED, YELLOW, DIM, END = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "min_bytes": 100_000,
        "why": "face detector (CPU)",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "emotion-ferplus-8.onnx": {
        "min_bytes": 10_000_000,
        "why": "expression estimation",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    },
    "centerface_dynamic.onnx": {
        "min_bytes": 5_000_000,
        "why": "face detector (GPU tier)",
        "url": "https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/"
               "models/onnx/centerface.onnx",
        "note": "shipped copy is patched for dynamic input; the download is "
                "the stock export and only works as a fallback on CPU",
    },
    "human_segmentation_pphumanseg_2023mar.onnx": {
        "min_bytes": 1_000_000,
        "why": "people-silhouette cutout (fast)",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/"
               "human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx",
    },
    "modnet_portrait.onnx": {
        "min_bytes": 5_000_000,
        "why": "people matte (quality — MODNet)",
        "url": "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
    },
    "rvm_mobilenetv3_fp32.onnx": {
        "min_bytes": 5_000_000,
        "why": "people matte (best — RVM, internal builds only)",
        "internal_only": True,
        "url": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/"
               "v1.0.0/rvm_mobilenetv3_fp32.onnx",
    },
}

_results: list[tuple[str, str, str]] = []  # (status, title, detail)


def _report(status: str, title: str, detail: str = "") -> None:
    _results.append((status, title, detail))
    mark = {"ok": f"{GREEN}  OK{END}", "warn": f"{YELLOW}WARN{END}", "fail": f"{RED}FAIL{END}"}[status]
    print(f" [{mark}] {title}" + (f"\n        {DIM}{detail}{END}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        _report("ok", f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _report("fail", f"Python {v.major}.{v.minor} is too old",
                "Install Python 3.10 or newer from python.org, then re-run Setup.")


def check_packages() -> None:
    core = ["numpy", "cv2", "cyndilib", "fastapi", "uvicorn", "websockets"]
    missing = []
    for mod in core:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _report("fail", f"Missing packages: {', '.join(missing)}",
                "Run the Setup script again (or: pip install -r requirements.txt).")
    else:
        _report("ok", "All required packages installed")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu = {"CUDAExecutionProvider", "TensorrtExecutionProvider"} & set(providers)
        if gpu:
            _report("ok", "NVIDIA GPU acceleration available",
                    "The CenterFace detector will be used automatically.")
        else:
            _report("ok", "onnxruntime installed (no NVIDIA GPU here)",
                    "The fast CPU detector (YuNet) will be used — normal on a Mac.")
    except ImportError:
        _report("warn", "onnxruntime not installed",
                "Only needed for the GPU detector; fine to ignore on a Mac.")


def _download(url: str, dest: Path, label: str) -> None:
    print(f"        {DIM}downloading {label}...{END}")
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def check_models(fix: bool) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    try:      # RVM ships in internal builds only (GPL-3.0) — see LICENSE
        from .edition import DISTRIBUTION
    except ImportError:
        DISTRIBUTION = False
    for name, info in MODELS.items():
        if DISTRIBUTION and info.get("internal_only"):
            continue
        path = MODELS_DIR / name
        if path.exists() and path.stat().st_size >= info["min_bytes"]:
            _report("ok", f"Model present: {name}", info["why"])
            continue
        if not fix:
            _report("fail", f"Model missing: {name} ({info['why']})",
                    "Run Setup again, or: python -m facetrack.doctor --fix")
            continue
        try:
            _download(info["url"], path, name)
            _report("ok", f"Model downloaded: {name}", info["why"])
        except Exception as exc:
            _report("fail", f"Could not download {name}", f"{exc} — check your internet connection.")


def check_camera() -> None:
    try:
        from .capture import camera_authorization, probe_cameras, request_camera_access
        request_camera_access()  # first run: pop the macOS prompt now
        cams = probe_cameras(max_index=4)
    except Exception as exc:
        _report("warn", "Camera check errored", str(exc))
        return
    if cams:
        _report("ok", f"Camera(s) found: {', '.join(c['label'] for c in cams)}")
        return
    auth = camera_authorization()
    if auth == "denied":
        _report("warn", "macOS is blocking camera access for this app",
                "System Settings > Privacy & Security > Camera — allow your "
                "terminal app, then run this check again.")
    elif auth in ("undetermined", "restricted"):
        _report("warn", "Camera permission not granted yet",
                "macOS should be showing a permission prompt — click Allow, "
                "then run this check again.")
    else:
        _report("warn", "No cameras detected",
                "Check the camera is connected and not in use by another app.")


def check_ndi() -> None:
    try:
        from fractions import Fraction
        from cyndilib.sender import Sender
        from cyndilib.video_frame import VideoSendFrame
        from cyndilib.wrapper.ndi_structs import FourCC
        vf = VideoSendFrame()
        vf.set_resolution(160, 90)
        vf.set_frame_rate(Fraction(30, 1))
        vf.set_fourcc(FourCC.BGRA)
        s = Sender(ndi_name="FACETRACK-DOCTOR")
        s.set_video_frame(vf)
        s.open()
        s.close()
        _report("ok", "NDI output works", "Receivers will see this machine on the network.")
    except Exception as exc:
        _report("fail", "NDI output failed", str(exc))


def check_texture_share() -> None:
    try:
        from .texture_out import probe
        kind, err = probe()
    except Exception as exc:
        kind, err = "", str(exc)
    if kind:
        _report("ok", f"Texture share available ({kind.capitalize()})",
                "Resolume / VDMX / TouchDesigner on this machine can take the feed directly.")
    else:
        _report("warn", "Texture share (Syphon/Spout) unavailable", err)


def check_port(port: int = 8089) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        _report("ok", f"Control panel port {port} is free")
    except OSError:
        _report("warn", f"Port {port} is in use",
                "facetrack may already be running — or launch with --web-port <other>.")
    finally:
        sock.close()


def main(argv=None) -> int:
    if sys.platform == "win32":
        import os
        os.system("")  # switches legacy consoles into ANSI-colour mode
    ap = argparse.ArgumentParser(description="facetrack self-check")
    ap.add_argument("--fix", action="store_true", help="download any missing model files")
    ap.add_argument("--no-camera", action="store_true", help="skip the camera probe")
    args = ap.parse_args(argv)

    print("\nfacetrack self-check\n" + "-" * 40)
    check_python()
    check_packages()
    check_models(args.fix)
    if not args.no_camera:
        check_camera()
    check_ndi()
    check_texture_share()
    check_port()
    print("-" * 40)

    fails = [r for r in _results if r[0] == "fail"]
    warns = [r for r in _results if r[0] == "warn"]
    if fails:
        print(f"{RED}{len(fails)} problem(s) need fixing before the show.{END}\n")
        return 1
    if warns:
        print(f"{YELLOW}Ready, with {len(warns)} note(s) above.{END}\n")
    else:
        print(f"{GREEN}Everything looks good.{END}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
