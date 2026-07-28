"""Where yewee reads and writes.

Running from a source checkout, settings and logs sit next to the code —
convenient while developing. Inside a packaged app they must not: an
app bundle is read-only by convention, macOS code signatures cover its
contents (writing into it invalidates the signature), and on Windows it
usually lives somewhere the user cannot write. Frozen builds therefore
keep everything in the per-user data directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PRODUCT = "yewee"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def user_data_dir() -> Path:
    """Per-user state that survives a reinstall."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / PRODUCT
    d.mkdir(parents=True, exist_ok=True)
    return d


def _source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    """Base for settings and logs: the checkout, or the user's data dir."""
    return user_data_dir() if is_frozen() else _source_root()


def settings_path() -> Path:
    return state_dir() / "settings.json"


def log_dir() -> Path:
    return state_dir() / "logs"


def log_path() -> Path:
    return log_dir() / f"{PRODUCT}.log"
