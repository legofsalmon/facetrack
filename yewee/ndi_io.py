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

    #: Turn the outgoing picture upside down. Unlike texture share there is
    #: no flag for it in NDI, so this costs a real copy — hence it only
    #: happens when switched on. Set live by the pipeline.
    flip = False

    def __init__(self, name: str = "Yewee", fps: float = 30.0):
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
        if self.flip:
            # New array, never in-place: the caller shares this frame with
            # the texture feeds, which have their own flip setting.
            frame = cv2.flip(frame, 0)
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

    #: Seconds without a new frame from the sender before read() stops
    #: returning one. NDI's frame sync never runs dry: when a sender goes
    #: away it repeats the last frame it had, for ever, so a frame coming
    #: back says nothing about whether the source is still there. Generous
    #: enough for a sender running at a few frames a second.
    STALE_AFTER = 2.0

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
        self._last_ts = None        # timestamp of the newest frame seen
        self._ts_at = 0.0           # when it arrived (monotonic)
        self._ts_changes = 0        # how often it has changed (capped)

    def _sender_alive(self, now: float) -> bool:
        """False once the sender has gone: no connection, or the same frame
        repeated for STALE_AFTER seconds. Call after capture_video()."""
        if not self.receiver.is_connected():
            return False
        ts = self.video_frame.get_timestamp_posix()
        if ts != self._last_ts:
            self._ts_changes = min(self._ts_changes + 1, 3)
            self._last_ts, self._ts_at = ts, now
            return True
        # Before the first frame the timestamp reads 0, and a sender that
        # doesn't stamp its frames sends one constant value after that. Only
        # a timestamp seen moving from frame to frame (a second change) can
        # go stale; for the others the connection is all there is to go on.
        return self._ts_changes < 3 or now - self._ts_at < self.STALE_AFTER

    def read(self, timeout: float = 5.0):
        """Returns (ok, frame_bgr); (False, None) when no frame arrives
        within timeout, including when the sender has gone and the frame
        sync is only repeating its last picture."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.receiver.frame_sync.capture_video()
            if not self._sender_alive(time.monotonic()):
                time.sleep(0.02)
                continue
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
