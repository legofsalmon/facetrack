"""Thread-safe live parameters shared between the pipeline and the web UI.

Every value here can change mid-run: the pipeline takes a snapshot() each
frame, and the web UI writes via set(), which validates and clamps.
"""
from __future__ import annotations

import threading

# name -> (type, min, max); None bounds for bools
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
    "clean_main": (bool, None, None),
    "flip": (bool, None, None),
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
