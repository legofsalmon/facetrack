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
import zipfile
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
    "scrfd_10g.onnx": {
        "min_bytes": 5_000_000,
        "why": "face detector (NVIDIA GPU)",
        "zip_url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "zip_member": "det_10g.onnx",
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
                    "The high-accuracy SCRFD detector will be used automatically.")
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
    for name, info in MODELS.items():
        path = MODELS_DIR / name
        if path.exists() and path.stat().st_size >= info["min_bytes"]:
            _report("ok", f"Model present: {name}", info["why"])
            continue
        if not fix:
            _report("fail", f"Model missing: {name} ({info['why']})",
                    "Run Setup again, or: python -m facetrack.doctor --fix")
            continue
        try:
            if "url" in info:
                _download(info["url"], path, name)
            else:  # inside a zip (SCRFD ships in the InsightFace pack, ~280 MB)
                zip_tmp = MODELS_DIR / "_pack.zip"
                _download(info["zip_url"], zip_tmp, "InsightFace model pack (~280 MB)")
                with zipfile.ZipFile(zip_tmp) as z:
                    member = next(m for m in z.namelist()
                                  if m.endswith(info["zip_member"]))
                    path.write_bytes(z.read(member))
                zip_tmp.unlink()
            _report("ok", f"Model downloaded: {name}", info["why"])
        except Exception as exc:
            _report("fail", f"Could not download {name}", f"{exc} — check your internet connection.")


def check_camera() -> None:
    try:
        from .capture import probe_cameras
        cams = probe_cameras(max_index=4)
    except Exception as exc:
        _report("warn", "Camera check errored", str(exc))
        return
    if cams:
        _report("ok", f"Camera(s) found: {', '.join(c['label'] for c in cams)}")
    else:
        detail = ("No camera opened. If this machine has one, grant camera permission: "
                  "System Settings > Privacy & Security > Camera (macOS) — the app asks "
                  "on first start." if sys.platform == "darwin"
                  else "No camera opened — check it is connected and not in use.")
        _report("warn", "No cameras detected", detail)


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
