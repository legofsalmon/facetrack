"""Thread-safe live parameters shared between the pipeline and the web UI.

Every value here can change mid-run: the pipeline takes a snapshot() each
frame, and the web UI writes via set(), which validates and clamps.
"""
from __future__ import annotations

import threading

# name -> (type, min, max); None bounds for bools; for str params the
# second slot is the tuple of allowed choices (first = fallback).
SPEC = {
    "det_threshold": (float, 0.05, 0.95),
    "det_size": (int, 160, 1920),
    "detect_every": (int, 1, 6),
    "min_face": (int, 0, 300),
    "max_misses": (int, 1, 90),
    "emotion_enabled": (bool, None, None),
    "emotion_budget": (int, 0, 16),
    "show_ids": (bool, None, None),
    "show_stats": (bool, None, None),
    "overlay_color": (str, None, None),   # "#rrggbb" brand colour; "" = palette
    "clean_main": (bool, None, None),
    "flip": (bool, None, None),
    # Capture negotiation — applied when a source is (re)opened. Format
    # is "WxH@fps" or "auto"; backend matters for capture cards (vendor
    # DirectShow filters on Windows, e.g. Blackmagic WDM).
    "cap_format": (str, None, None),
    "cap_backend": (str, ("any", "avfoundation", "dshow", "msmf"), None),
    # The output matrix: four content types x two transports. NDI feeds
    # are named "<name>", "<name> Overlay/Faces/Mask"; texture servers
    # (Syphon/Spout) are "facetrack", "facetrack-overlay/-faces/-mask".
    "ndi_program": (bool, None, None),
    "ndi_overlay": (bool, None, None),
    "ndi_faces": (bool, None, None),
    "ndi_mask": (bool, None, None),
    "tex_program": (bool, None, None),
    "tex_overlay": (bool, None, None),
    "tex_faces": (bool, None, None),
    "tex_mask": (bool, None, None),
    "mask_style": (str, ("white", "alpha"), None),  # luma matte / on alpha
    "out_width": (int, 0, 3840),          # 0 = match input resolution
    "cutout_shape": (str, ("rectangle", "oval", "people"), None),
    "cutout_margin": (float, 0.0, 0.5),   # extra room around each face box
    "cutout_feather": (int, 0, 60),       # mask edge softness, px
    "cutout_steady": (float, 0.0, 0.9),   # people-mask temporal smoothing
    "people_model": (str, ("pphumanseg", "modnet", "rvm"), None),
    "test_card": (bool, None, None),      # bars + motion on all feeds
    "panel_preview": (bool, None, None),  # MJPEG thumbnail in the web panel
    "preview_source": (str, ("annotated", "clean", "overlay", "faces", "mask"), None),
    "local_preview": (bool, None, None),  # preview window on the machine
}


class LiveParams:
    def __init__(self, **initial):
        self._lock = threading.Lock()
        self._values = {}
        for key in SPEC:
            if key not in initial:
                raise KeyError(f"missing initial value for {key}")
            self._values[key] = self._coerce(key, initial[key])

    @staticmethod
    def _coerce(key: str, value):
        typ, lo, hi = SPEC[key]
        if typ is bool:
            return bool(value)
        if typ is str:
            if lo is None:  # free-form string (e.g. a colour), length-capped
                return str(value).strip()[:32]
            v = str(value)
            return v if v in lo else lo[0]
        v = typ(value)
        return max(lo, min(hi, v))

    def set(self, key: str, value):
        if key not in SPEC:
            raise KeyError(key)
        v = self._coerce(key, value)
        with self._lock:
            self._values[key] = v
        return v

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._values)
