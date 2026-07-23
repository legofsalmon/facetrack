"""NDI output (and optional NDI input) via cyndilib.

The NDI runtime library is bundled with the cyndilib wheels, so no separate
NDI SDK install is required on either macOS or Windows.
"""
from __future__ import annotations

import time
from fractions import Fraction

import cv2
import numpy as np

from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC


class NDIOutput:
    """Sends BGR frames as an NDI video source. Handles resolution changes
    by transparently re-opening the sender."""

    def __init__(self, name: str = "FaceTracker", fps: float = 30.0):
        self.name = name
        self.fps = float(fps)
        self.sender: Sender | None = None
        self.size: tuple[int, int] | None = None
        self._hold = None  # keep the async buffer alive until the next send

    def _reopen(self, w: int, h: int) -> None:
        if self.sender is not None:
            self.sender.close()
        vf = VideoSendFrame()
        vf.set_resolution(w, h)
        vf.set_frame_rate(Fraction(self.fps).limit_denominator(60000))
        vf.set_fourcc(FourCC.BGRA)
        sender = Sender(ndi_name=self.name, clock_video=False, clock_audio=False)
        sender.set_video_frame(vf)
        sender.open()
        self.sender = sender
        self.size = (w, h)

    def send(self, frame: np.ndarray) -> None:
        """Accepts BGR (opaque) or BGRA (e.g. overlay-on-transparency)."""
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self._reopen(w, h)
        if frame.shape[2] == 3:
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        else:
            bgra = frame
        if not bgra.flags["C_CONTIGUOUS"]:
            bgra = np.ascontiguousarray(bgra)
        self.sender.write_video_async(bgra.reshape(-1))
        self._hold = bgra

    def close(self) -> None:
        if self.sender is not None:
            self.sender.close()
            self.sender = None


class NDIInput:
    """Receives an NDI source as BGR frames (so the tracker can sit
    anywhere in an existing NDI chain)."""

    def __init__(self, source_name: str, timeout: float = 10.0):
        from cyndilib.finder import Finder
        from cyndilib.receiver import Receiver
        from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
        from cyndilib.video_frame import VideoFrameSync

        self.finder = Finder()
        self.finder.open()
        deadline = time.monotonic() + timeout
        source = None
        wanted = source_name.lower()
        while time.monotonic() < deadline and source is None:
            self.finder.wait_for_sources(timeout=1.0)
            for name in self.finder.get_source_names():
                if wanted in name.lower():
                    source = self.finder.get_source(name)
                    break
        if source is None:
            names = list(self.finder.get_source_names())
            self.finder.close()
            raise RuntimeError(
                f"NDI source matching {source_name!r} not found. Visible sources: {names or 'none'}")

        self.receiver = Receiver(color_format=RecvColorFormat.BGRX_BGRA,
                                 bandwidth=RecvBandwidth.highest)
        self.video_frame = VideoFrameSync()
        self.receiver.frame_sync.set_video_frame(self.video_frame)
        self.receiver.set_source(source)
        self.source_display_name = str(source.name)

    def read(self, timeout: float = 5.0):
        """Returns (ok, frame_bgr)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.receiver.frame_sync.capture_video()
            xres, yres = self.video_frame.xres, self.video_frame.yres
            if xres > 0 and yres > 0:
                # View the frame buffer, convert (copies), then drop the view:
                # cyndilib refuses the next capture while a view is alive.
                data = np.asarray(self.video_frame)
                expected = yres * xres * 4
                if data.size < expected:
                    time.sleep(0.005)
                    continue
                if data.size > expected:  # padded line stride
                    stride = data.size // yres
                    view = data.reshape(yres, stride)[:, :xres * 4].reshape(yres, xres, 4)
                else:
                    view = data.reshape(yres, xres, 4)
                bgr = cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)
                del view, data
                return True, bgr
            time.sleep(0.005)
        return False, None

    def close(self) -> None:
        try:
            self.receiver.disconnect()
        except Exception:
            pass
        self.finder.close()
