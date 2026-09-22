"""NDI output (and optional NDI input) via cyndilib.

The NDI runtime library is bundled with the cyndilib wheels, so no separate
NDI SDK install is required on either macOS or Windows.
"""
from __future__ import annotations

import logging
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
    anywhere in an existing NDI chain).

    The frame sync holds the latest frame and hands it back on demand,
    new or not, so a naive read() returns duplicates as fast as it is
    called. read() therefore waits for the frame's own stamp to move,
    which makes an NDI input pace the pipeline at the rate it really
    sends — a 50 or 60 Hz feed is no longer processed at 30.

    Which stamp field cyndilib surfaces varies, and one that never moves
    would wedge the feed, so patience is bounded: if the stamp has not
    changed within `NOVELTY_WAIT` the receiver is marked unstamped for
    good, `stamped` goes False, and the pipeline goes back to pacing the
    source at its declared rate.
    """

    #: NDI carries both; take whichever this build of cyndilib exposes.
    STAMP_FIELDS = ("timestamp", "timecode")
    #: How long to wait for a new frame before giving up on stamps (s).
    NOVELTY_WAIT = 0.5

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
        self._stamp_field = self._find_stamp_field()
        self.stamped = self._stamp_field is not None
        self._last_stamp = None

    def _find_stamp_field(self) -> str | None:
        """The frame attribute that identifies one frame, or None."""
        for name in self.STAMP_FIELDS:
            try:
                if getattr(self.video_frame, name) is not None:
                    return name
            except Exception:
                continue
        return None

    def _stamp(self):
        try:
            return int(getattr(self.video_frame, self._stamp_field))
        except Exception:
            return None

    def read(self, timeout: float = 5.0):
        """Returns (ok, frame_bgr) for a frame not already returned."""
        deadline = time.monotonic() + timeout
        stale_until = time.monotonic() + self.NOVELTY_WAIT
        while time.monotonic() < deadline:
            self.receiver.frame_sync.capture_video()
            xres, yres = self.video_frame.xres, self.video_frame.yres
            if xres > 0 and yres > 0:
                if self.stamped:
                    stamp = self._stamp()
                    if stamp is not None and stamp == self._last_stamp:
                        if time.monotonic() < stale_until:
                            time.sleep(0.002)
                            continue
                        # the stamp is not moving — it is decorative on this
                        # receiver, so stop trusting it and let the pipeline
                        # pace us again rather than starving the show
                        self.stamped = False
                        logging.getLogger("yewee").info(
                            "NDI input exposes no moving frame stamp — "
                            "falling back to paced reads")
                    else:
                        self._last_stamp = stamp
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
