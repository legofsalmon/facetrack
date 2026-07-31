"""Frame sources: camera / capture card (threaded, latest-frame-wins),
video file (sequential), or NDI input.

Camera capture runs in its own thread so a slow processing frame never
backs up the driver queue — the pipeline always gets the freshest frame,
which keeps end-to-end latency low.
"""
from __future__ import annotations

import logging
import os
import threading
import time

# OpenCV's AVFoundation backend asks for camera permission itself, from
# whichever thread opens the device — and ours is a worker thread, so the
# request cannot work (the prompt needs the main run loop) and it logs a
# baffling error instead. request_camera_access() asks properly at startup,
# so tell OpenCV not to try. Must be set before any capture is opened.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2  # noqa: E402  (after the env var above, which cv2 reads on open)

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


def camera_permission_holder() -> str:
    """What the user has to allow in System Settings > Camera. A packaged
    app holds the permission itself; from source it is granted to whatever
    launched us, which is normally a terminal."""
    from .paths import is_frozen
    return "Yewee" if is_frozen() else "your terminal app"


def request_camera_access(timeout: float = 45.0) -> str:
    """Asks macOS for camera permission and waits for the answer.

    Call this on the main thread before anything opens a camera. Both
    halves matter: the prompt only appears while the main run loop is
    turning, and the request is asynchronous, so without waiting we would
    race ahead and open the camera before the user had answered — which is
    exactly how a first launch fails with a bare 'Could not open camera'.

    Returns the resulting authorisation state. No-op off macOS or once the
    question has already been settled — so this costs nothing after the
    first launch. The timeout is a backstop against the app looking hung
    if nobody is at the keyboard; permission granted later still takes
    effect, it just needs the source re-picked in the panel.
    """
    import sys
    if sys.platform != "darwin":
        return "authorized"
    if camera_authorization() != "undetermined":
        return camera_authorization()
    try:
        import objc
        from Foundation import NSDate, NSRunLoop
        objc.loadBundle("AVFoundation", {},
                        bundle_path="/System/Library/Frameworks/AVFoundation.framework")
        # The completion handler is an Objective-C block, and PyObjC will not
        # pass a Python callable as one without knowing its signature — it
        # raises "no signature available". The proper AVFoundation bindings
        # carry this metadata; we register just the one selector we need
        # rather than ship the whole framework wrapper.
        objc.registerMetaDataForSelector(
            b"AVCaptureDevice", b"requestAccessForMediaType:completionHandler:",
            {"arguments": {3: {"callable": {
                "retval": {"type": b"v"},
                "arguments": {0: {"type": b"^v"}, 1: {"type": objc._C_BOOL}}}}}})

        dev = objc.lookUpClass("AVCaptureDevice")
        answered: list[bool] = []
        dev.requestAccessForMediaType_completionHandler_(
            "vide", lambda ok: answered.append(bool(ok)))

        # Wait on the recorded status rather than only the callback: the
        # status is what every later camera open actually consults, and it
        # keeps us honest if the block ever stops firing. Pumping the run
        # loop is what lets the prompt draw in the first place.
        loop = NSRunLoop.currentRunLoop()
        deadline = time.monotonic() + timeout
        while (not answered and camera_authorization() == "undetermined"
               and time.monotonic() < deadline):
            loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))
    except Exception as exc:
        # Worth saying out loud: silently swallowing this is how the request
        # went unnoticed as broken, leaving users with a camera that could
        # never be authorised.
        logging.getLogger("yewee").warning(
            "could not ask for camera permission: %s", exc)
    return camera_authorization()


def _camera_names() -> list[str]:
    """Device names in OpenCV's index order — from the same API OpenCV
    enumerates with, because no other list is safe to pair with its
    indices. The names used to come from system_profiler, whose order is
    unstable and unrelated: when the two lists crossed, the entry labelled
    "Blackmagic UltraStudio" opened the FaceTime camera, with no error
    anywhere because a camera did open. Windows pairs pygrabber with
    CAP_DSHOW the same way (both walk DirectShow filters in system order).
    """
    import sys
    try:
        if sys.platform == "darwin":
            import objc
            objc.loadBundle(
                "AVFoundation", {},
                bundle_path="/System/Library/Frameworks/AVFoundation.framework")
            # Mirror OpenCV's cap_avfoundation_mac.mm exactly: video devices,
            # then muxed devices appended, then the whole list SORTED BY
            # uniqueID — that sort is the part nobody expects, and it is why
            # the raw enumeration (whose order genuinely wobbles between
            # calls) still opens the same camera at the same index every
            # time in OpenCV. Skip any step, including keeping duplicates
            # when a device appears under both media types, and the indices
            # here stop being OpenCV's indices.
            dev = objc.lookUpClass("AVCaptureDevice")
            devs = list(dev.devicesWithMediaType_("vide")) \
                + list(dev.devicesWithMediaType_("muxe"))
            devs.sort(key=lambda d: str(d.uniqueID()))
            return [str(d.localizedName()) for d in devs]
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
        name = names[i] if i < len(names) and names[i] else ""
        # Select by name whenever the OS gave us one: the index is only
        # valid until the enumeration order shifts, and it does shift.
        spec = f"cam:{name}" if name else str(i)
        label = name or f"Camera {i}"
        if skip is not None and i == skip:
            found.append({"index": i, "spec": spec, "label": f"{label} (in use)"})
            misses = 0
            continue
        cap = cv2.VideoCapture(i, BACKENDS.get(backend, cv2.CAP_ANY))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append({"index": i, "spec": spec, "label": f"{label} ({w}x{h})"})
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


def resolve_camera(name: str) -> int:
    """Find the current index of a camera by device name.

    Camera indices are positions in an enumeration whose order genuinely
    changes — between processes, and when devices are plugged or unplugged.
    A saved index that meant the capture card yesterday can mean the
    built-in camera today, which on a live output is not a small mistake.
    The device name is the only stable handle, so it is resolved against a
    fresh enumeration every time a camera is opened, never cached.
    """
    names = _camera_names()
    if name in names:
        return names.index(name)
    want = name.strip().lower()
    loose = [i for i, n in enumerate(names) if want and want in n.lower()]
    if len(loose) == 1:
        return loose[0]
    raise RuntimeError(
        f"camera '{name}' not found — connected now: {names or 'none'}")


def open_source(spec: str, width: int = 0, height: int = 0, fps: float = 0.0,
                backend: str = "any", loop: bool = False):
    """spec: camera by name ('cam:<device name>'), camera index ('0'),
    video file/URL, or 'ndi:<source name>'. Prefer cam: — an index is only
    meaningful within one enumeration, a name survives replugs and reboots.
    """
    if spec.lower().startswith("ndi:"):
        return NDISource(spec[4:])
    if spec.lower().startswith("cam:"):
        return CameraSource(resolve_camera(spec[4:]), width, height, fps, backend)
    if spec.isdigit():
        return CameraSource(int(spec), width, height, fps, backend)
    return FileSource(spec, loop=loop)
