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


def load() -> dict:
    """Returns {"params": {...only known keys...}, "source": str|None}."""
    try:
        data = json.loads(SETTINGS_PATH.read_text())
    except (OSError, ValueError):
        return {"params": {}, "source": None}
    params = data.get("params", {})
    return {
        "params": {k: v for k, v in params.items() if k in SPEC},
        "source": data.get("source") or None,
    }


def _write(update: dict) -> None:
    with _lock:
        current = load()
        if "params" in update:
            current["params"].update(update["params"])
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
