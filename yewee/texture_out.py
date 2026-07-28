"""GPU texture-share output: Syphon (macOS) / Spout (Windows).

Publishes frames directly to VJ/media-server apps on the same machine
(Resolume, VDMX, MadMapper, TouchDesigner...) with no network hop and no
NDI compression — including real alpha, so the overlay-only mode keys
perfectly.

Availability is platform + Python dependent (syphon-python needs
Python <= 3.12, SpoutGL <= 3.13; the Setup scripts pick a compatible
Python automatically). probe() reports (kind, error) without side effects;
create() returns a ready output or raises RuntimeError with a fix hint.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

GL_RGBA = 0x1908  # literal so we don't need PyOpenGL for one constant


def probe() -> tuple[str, str]:
    """Returns (kind, error): kind is 'syphon', 'spout' or ''; error is a
    human fix hint when unavailable."""
    if sys.platform == "darwin":
        try:
            import syphon  # noqa: F401
            return "syphon", ""
        except ImportError:
            return "", ("Syphon needs the syphon-python package (Python 3.12) — "
                        "re-run Setup to fix.")
    if sys.platform == "win32":
        try:
            import SpoutGL  # noqa: F401
            return "spout", ""
        except ImportError:
            return "", ("Spout needs the SpoutGL package (Python 3.13 or older) — "
                        "re-run Setup to fix.")
    return "", "Texture share is only available on macOS (Syphon) and Windows (Spout)."


class SyphonOutput:
    kind = "syphon"

    def __init__(self, name: str = "yewee"):
        import syphon
        from Cocoa import NSDate, NSDefaultRunLoopMode, NSRunLoop
        from syphon.utils.numpy import copy_image_to_mtl_texture
        from syphon.utils.raw import create_mtl_texture
        self._create_texture = create_mtl_texture
        self._copy = copy_image_to_mtl_texture
        self._runloop = NSRunLoop.currentRunLoop()
        self._runloop_mode = NSDefaultRunLoopMode
        self._past = NSDate.distantPast()
        self.server = syphon.SyphonMetalServer(name)
        self._texture = None
        self._size: tuple[int, int] | None = None

    def send(self, frame: np.ndarray) -> None:
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA if frame.shape[2] == 3
                            else cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        if self._size != (w, h):
            self._texture = self._create_texture(self.server.device, w, h)
            self._size = (w, h)
        self._copy(rgba, self._texture)
        # is_flipped: our frames are top-down (OpenCV) while Syphon's
        # convention is GL bottom-up — without this flag receivers show
        # the picture upside down.
        self.server.publish_frame_texture(self._texture, is_flipped=True)
        # Service the run loop so Syphon's discovery handshake (distributed
        # notifications) works — otherwise apps started after us never see
        # the server.
        self._runloop.runMode_beforeDate_(self._runloop_mode, self._past)

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass


class SpoutOutput:
    kind = "spout"

    def __init__(self, name: str = "yewee"):
        import SpoutGL
        self.sender = SpoutGL.SpoutSender()
        self.sender.setSenderName(name)

    def send(self, frame: np.ndarray) -> None:
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA if frame.shape[2] == 3
                            else cv2.COLOR_BGRA2RGBA)
        h, w = rgba.shape[:2]
        if not rgba.flags["C_CONTIGUOUS"]:
            rgba = np.ascontiguousarray(rgba)
        # bInvert=True for the same reason as Syphon's is_flipped: the
        # buffer is top-down, GL/Spout convention is bottom-up.
        self.sender.sendImage(rgba.tobytes(), w, h, GL_RGBA, True, 0)

    def close(self) -> None:
        try:
            self.sender.releaseSender()
        except Exception:
            pass


def create(name: str = "yewee"):
    kind, err = probe()
    if not kind:
        raise RuntimeError(err)
    return SyphonOutput(name) if kind == "syphon" else SpoutOutput(name)
