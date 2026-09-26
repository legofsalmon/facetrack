"""Automatic settings persistence.

Panel changes are saved to settings.json (next to main.py) and restored on
the next launch, so operators never need CLI flags. Explicit CLI flags
still win for a single run. Writes are atomic (tmp + rename) and debounced
so slider drags don't hammer the disk.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path

from .params import SPEC

from .paths import settings_path

SETTINGS_PATH = settings_path()

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
    pin = "" if data.get("pin") is None else str(data.get("pin")).strip()
    return {
        "params": known,
        "source": data.get("source") or None,
        # the raw saved value; panel_pin() decides what it means
        "pin": pin,
    }


def _write(update: dict) -> None:
    with _lock:
        current = _read_raw()  # keep keys this write does not touch
        if "params" in update:
            merged = current.get("params", {})
            merged.update(update["params"])
            current["params"] = merged
        for key, value in update.items():
            if key != "params":
                current[key] = value     # "source", "pin", or a section such as "reports"
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(current, indent=2))
            os.replace(tmp, SETTINGS_PATH)
        except OSError:
            pass  # persistence is best-effort; never break the show over it


#: What settings.json holds (and --pin takes) when the operator has turned
#: the panel PIN off on purpose. A missing or empty "pin" is not "off": it
#: is a first run, and gets a new PIN.
PIN_OFF = "none"
PIN_OFF_WORDS = {"none", "off"}


def new_pin() -> str:
    """Four random digits from the OS's CSPRNG (not random.random)."""
    return f"{secrets.randbelow(10_000):04d}"


def panel_pin(cli: str | None = None) -> tuple[str, str]:
    """The PIN other devices must give the control panel, "" for none, and
    where it came from ("cli", "saved", "new" or "off").

    - ``--pin 4721`` sets the PIN and keeps it for next time;
    - ``--pin none`` turns it off and keeps that (saved as "pin": "none");
    - otherwise the PIN saved in settings.json;
    - otherwise, on a first run, four random digits, saved.

    A write that fails (read-only disk) still protects this run with the
    PIN; the next launch just makes another one."""
    if cli is not None and cli.strip():
        value = cli.strip()
        if value.lower() in PIN_OFF_WORDS:
            _write({"pin": PIN_OFF})
            return "", "off"
        _write({"pin": value})
        return value, "cli"
    raw = _read_raw().get("pin")
    saved = "" if raw is None else str(raw).strip()
    if saved.lower() in PIN_OFF_WORDS:
        return "", "off"
    if saved:
        return saved, "saved"
    value = new_pin()
    _write({"pin": value})
    return value, "new"


def save(params: dict | None = None, source: str | None = None) -> None:
    update: dict = {}
    if params is not None:
        update["params"] = dict(params)
    if source is not None:
        update["source"] = source
    if update:
        _write(update)


def load_section(name: str) -> dict:
    """A top-level section of settings.json other than params/source, e.g.
    "reports" (crash-report consent and the install id)."""
    value = _read_raw().get(name)
    return dict(value) if isinstance(value, dict) else {}


def save_section(name: str, values: dict) -> None:
    if name in ("params", "source", "pin"):
        raise ValueError(f"{name} is not a free section")
    _write({name: dict(values)})


def save_debounced(params: dict, delay: float = 0.6) -> None:
    global _timer, _pending
    _pending = dict(params)
    if _timer is not None:
        _timer.cancel()
    _timer = threading.Timer(delay, lambda: _write({"params": _pending}))
    _timer.daemon = True
    _timer.start()
