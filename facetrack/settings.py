"""Automatic settings persistence.

Panel changes are saved to settings.json (next to main.py) and restored on
the next launch, so operators never need CLI flags. Explicit CLI flags
still win for a single run. Writes are atomic (tmp + rename) and debounced
so slider drags don't hammer the disk.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .params import SPEC

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

_lock = threading.Lock()
_timer: threading.Timer | None = None
_pending: dict = {}


def _read_raw() -> dict:
    try:
        data = json.loads(SETTINGS_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load() -> dict:
    """Returns {"params": {...only known keys...}, "source": str|None,
    "pin": str}. Unknown top-level keys are preserved by writes."""
    data = _read_raw()
    params = data.get("params", {})
    known = {k: v for k, v in params.items() if k in SPEC}
    # migrations from before the output matrix (July 2026)
    if "ndi_program" not in known and "ndi_main" in params:
        known["ndi_program"] = bool(params["ndi_main"])
    if params.get("texture_share") and not any(
            k in known for k in ("tex_program", "tex_overlay", "tex_faces")):
        src = params.get("texture_source",
                         "overlay" if params.get("texture_overlay") else "program")
        known[{"program": "tex_program", "overlay": "tex_overlay",
               "faces": "tex_faces"}.get(src, "tex_program")] = True
    return {
        "params": known,
        "source": data.get("source") or None,
        "pin": str(data.get("pin") or ""),
    }


def _write(update: dict) -> None:
    with _lock:
        current = _read_raw()  # keep unknown keys (e.g. a hand-added "pin")
        if "params" in update:
            merged = current.get("params", {})
            merged.update(update["params"])
            current["params"] = merged
        if "source" in update:
            current["source"] = update["source"]
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(current, indent=2))
            os.replace(tmp, SETTINGS_PATH)
        except OSError:
            pass  # persistence is best-effort; never break the show over it


def save(params: dict | None = None, source: str | None = None) -> None:
    update: dict = {}
    if params is not None:
        update["params"] = dict(params)
    if source is not None:
        update["source"] = source
    if update:
        _write(update)


def save_debounced(params: dict, delay: float = 0.6) -> None:
    global _timer, _pending
    _pending = dict(params)
    if _timer is not None:
        _timer.cancel()
    _timer = threading.Timer(delay, lambda: _write({"params": _pending}))
    _timer.daemon = True
    _timer.start()
