"""Frame sources: camera / capture card (threaded, latest-frame-wins),
video file (sequential), or NDI input.

Camera capture runs in its own thread so a slow processing frame never
backs up the driver queue — the pipeline always gets the freshest frame,
which keeps end-to-end latency low.
"""
from __future__ import annotations

import threading
import time

import cv2

BACKENDS = {
    "any": cv2.CAP_ANY,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


class CameraSource:
    is_live = True

    def __init__(self, index: int, width: int = 0, height: int = 0,
                 fps: float = 0.0, backend: str = "any"):
        self.cap = cv2.VideoCapture(index, BACKENDS.get(backend, cv2.CAP_ANY))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0):
        """Blocks until a frame newer than the last returned one arrives."""
        with self._cond:
            seq = self._seq
            if not self._cond.wait_for(lambda: self._seq > seq, timeout=timeout):
                return False, None
            return True, self._frame

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class FileSource:
    is_live = False

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file/URL: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self, timeout: float = 0.0):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        self.cap.release()


class NDISource:
    is_live = True

    def __init__(self, source_name: str):
        from .ndi_io import NDIInput
        self.ndi = NDIInput(source_name)
        self.fps = 30.0

    def read(self, timeout: float = 5.0):
        return self.ndi.read(timeout=timeout)

    def close(self) -> None:
        self.ndi.close()


class NullSource:
    """Emits a 'no input' slate so the app (and its control panel) always
    starts, even when the configured source is unavailable."""

    is_live = True

    def __init__(self, width: int = 1280, height: int = 720, message: str = "NO INPUT"):
        import numpy as np
        self.fps = 15.0
        self._frame = np.zeros((height, width, 3), dtype="uint8")
        self._frame[:] = (24, 20, 18)
        for i, line in enumerate([message, "Choose a source in the control panel"]):
            scale = 1.4 if i == 0 else 0.7
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.putText(self._frame, line, ((width - tw) // 2, height // 2 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (140, 140, 150), 2, cv2.LINE_AA)

    def read(self, timeout: float = 0.0):
        time.sleep(1.0 / self.fps)
        return True, self._frame.copy()

    def close(self) -> None:
        pass


def camera_authorization() -> str:
    """macOS camera permission for THIS process: 'authorized', 'denied',
    'undetermined', 'restricted', or 'unknown'. Other platforms have no
    such gate and report 'authorized'."""
    import sys
    if sys.platform != "darwin":
        return "authorized"
    try:
        import objc
        objc.loadBundle("AVFoundation", {},
                        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        dev = objc.lookUpClass("AVCaptureDevice")
        status = int(dev.authorizationStatusForMediaType_("vide"))
        return {0: "undetermined", 1: "restricted",
                2: "denied", 3: "authorized"}.get(status, "unknown")
    except Exception:
        return "unknown"


def request_camera_access() -> None:
    """Triggers the macOS camera-permission prompt (no-op elsewhere or if
    already decided). The prompt is attributed to the app that launched us
    — normally the user's terminal."""
    import sys
    if sys.platform != "darwin" or camera_authorization() != "undetermined":
        return
    try:
        import objc
        objc.loadBundle("AVFoundation", {},
                        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        dev = objc.lookUpClass("AVCaptureDevice")
        dev.requestAccessForMediaType_completionHandler_("vide", lambda ok: None)
    except Exception:
        pass


def _camera_names() -> list[str]:
    """Best-effort device names in index order (macOS: system_profiler,
    Windows: DirectShow via pygrabber). Virtual cameras that the OS doesn't
    list still get probed — they just fall back to a generic label."""
    import sys
    try:
        if sys.platform == "darwin":
            import json
            import subprocess
            out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                                 capture_output=True, timeout=8)
            data = json.loads(out.stdout or b"{}")
            return [c.get("_name", "") for c in data.get("SPCameraDataType", [])]
        if sys.platform == "win32":
            from pygrabber.dshow_graph import FilterGraph
            return FilterGraph().get_input_devices()
    except Exception:
        pass
    return []


def probe_cameras(max_index: int = 8, backend: str = "any",
                  skip: int | None = None) -> list[dict]:
    """Checks which camera indices open, with real device names where the
    OS provides them. `skip` marks the index the pipeline is already using
    (reported without opening it, so the live capture isn't disturbed).
    Safe to call while running — the panel's rescan button uses this."""
    names = _camera_names()
    found = []
    misses = 0
    for i in range(max_index):
        name = names[i] if i < len(names) and names[i] else f"Camera {i}"
        if skip is not None and i == skip:
            found.append({"index": i, "label": f"{name} (in use)"})
            misses = 0
            continue
        cap = cv2.VideoCapture(i, BACKENDS.get(backend, cv2.CAP_ANY))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append({"index": i, "label": f"{name} ({w}x{h})"})
            misses = 0
        else:
            misses += 1
        cap.release()
        if misses >= 2 and i >= len(names):
            break  # two consecutive dead indices past the known devices
    return found


def parse_cap_format(fmt: str) -> tuple[int, int, float]:
    """'1920x1080@50' -> (1920, 1080, 50.0); 'auto'/invalid -> (0, 0, 0)."""
    try:
        size, fps = fmt.strip().lower().split("@")
        w, h = size.split("x")
        return int(w), int(h), float(fps)
    except (ValueError, AttributeError):
        return 0, 0, 0.0


def open_source(spec: str, width: int = 0, height: int = 0, fps: float = 0.0,
                backend: str = "any", loop: bool = False):
    """spec: camera index ('0'), video file/URL, or 'ndi:<source name>'."""
    if spec.lower().startswith("ndi:"):
        return NDISource(spec[4:])
    if spec.isdigit():
        return CameraSource(int(spec), width, height, fps, backend)
    return FileSource(spec, loop=loop)
