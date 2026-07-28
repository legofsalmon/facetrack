"""Licence keys and the trial clock.

Keys are Ed25519-signed blobs that the app verifies **offline** with an
embedded public key, so activation needs no server — the same key works
whether the machine is online or air-gapped, which covers both
activation paths and makes free reviewer keys trivial to issue.

    YW1.<base64url payload>.<base64url signature>

The payload is compact JSON:

    {"v":1, "p":"yewee", "e":"pro", "n":"Jane Smith",
     "i":"2026-07-28", "x":"2027-07-28", "m":"<machine>", "k":"<id>"}

`x` (expiry) and `m` (machine binding) are optional — a key without
either is perpetual and works on any machine, which is what a normal
one-off purchase gets. `k` is a key id, so a future server can revoke.

Enforcement only switches on when VENDOR_PUBLIC_KEY is set, which the
packaging step does for a distributed build. Repo and internal builds
leave it empty and run unrestricted — see yewee/edition.py.

Honest limitation: this is Python, so a determined user can edit the
check out. The goal is keeping honest people honest, not DRM.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from . import _ed25519 as ed

# Baked in by the packaging step (yewee/_buildinfo.py) for a sold build.
# A source checkout has neither, so licensing stays dormant and the app
# runs unrestricted — see edition.py.
try:
    from ._buildinfo import VENDOR_PUBLIC_KEY      # type: ignore
except ImportError:
    VENDOR_PUBLIC_KEY = os.environ.get("YEWEE_PUBKEY", "")

TRIAL_HOURS = 72
PRODUCT = "yewee"
_KEY_PREFIX = "YW1."


# ---------------------------------------------------------------- paths

def user_data_dir() -> Path:
    """Per-user state that survives a reinstall (so does the trial)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / PRODUCT
    d.mkdir(parents=True, exist_ok=True)
    return d


def _secondary_anchor() -> Path:
    """A second home for the trial clock, so deleting one file is not
    enough to reset it. Deliberately unremarkable."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Preferences"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return base / f".{PRODUCT}-id"


def machine_id() -> str:
    """Stable-ish per-machine fingerprint (hashed, never the raw serial)."""
    raw = ""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    raw = line.split('"')[-2]
                    break
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Cryptography") as k:
                raw = winreg.QueryValueEx(k, "MachineGuid")[0]
    except Exception:
        raw = ""
    if not raw:
        import uuid
        raw = str(uuid.getnode())
    return hashlib.sha256(f"{PRODUCT}:{raw}".encode()).hexdigest()[:16]


# ------------------------------------------------------------ key format

def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def encode_key(payload: dict, secret: bytes) -> str:
    """Vendor side: sign a payload into a key string."""
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return f"{_KEY_PREFIX}{_b64e(body)}.{_b64e(ed.sign(secret, body))}"


def decode_key(key: str, public_key_hex: str | None = None) -> dict | None:
    """Verify a key and return its payload, or None if it isn't valid."""
    pub_hex = VENDOR_PUBLIC_KEY if public_key_hex is None else public_key_hex
    if not pub_hex:
        return None
    key = "".join((key or "").split())
    if not key.startswith(_KEY_PREFIX):
        return None
    try:
        body_b64, sig_b64 = key[len(_KEY_PREFIX):].split(".", 1)
        body, sig = _b64d(body_b64), _b64d(sig_b64)
        if not ed.verify(bytes.fromhex(pub_hex), body, sig):
            return None
        payload = json.loads(body)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("p") != PRODUCT:
        return None
    return payload


# ------------------------------------------------------------- storage

def _licence_path() -> Path:
    return user_data_dir() / "licence.key"


def stored_key() -> str:
    try:
        return _licence_path().read_text().strip()
    except OSError:
        return ""


def activate(key: str) -> tuple[bool, str]:
    """Verify and store a key. Returns (ok, message for the operator)."""
    if not VENDOR_PUBLIC_KEY:
        return False, "This build does not use licence keys."
    payload = decode_key(key)
    if payload is None:
        return False, "That key isn't valid for yewee."
    bound = payload.get("m")
    if bound and bound != machine_id():
        return False, "That key is registered to a different machine."
    expiry = _expiry_date(payload)
    if expiry is not None and expiry < date.today():
        return False, f"That key expired on {expiry.isoformat()}."
    try:
        _licence_path().write_text("".join(key.split()))
    except OSError as exc:
        return False, f"Could not save the licence: {exc}"
    return True, f"Activated — thank you, {payload.get('n', 'friend')}."


def deactivate() -> None:
    try:
        _licence_path().unlink()
    except OSError:
        pass


def _expiry_date(payload: dict):
    raw = payload.get("x")
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


# --------------------------------------------------------- trial clock

def _read_anchor(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _trial_state() -> dict:
    """First-run and last-seen times, taking the *earliest* first run any
    anchor knows about so removing one file doesn't restart the trial."""
    mid = machine_id()
    anchors = [user_data_dir() / "trial.json", _secondary_anchor()]
    first, last = None, 0.0
    for path in anchors:
        data = _read_anchor(path)
        if data.get("machine") != mid:
            continue
        if isinstance(data.get("first"), (int, float)):
            first = data["first"] if first is None else min(first, data["first"])
        if isinstance(data.get("last"), (int, float)):
            last = max(last, data["last"])
    now = time.time()
    if first is None:
        first = now
    # a rolled-back clock must not hand back trial time
    now = max(now, last)
    for path in anchors:
        try:
            path.write_text(json.dumps({"machine": mid, "first": first, "last": now}))
        except OSError:
            pass
    return {"first": first, "now": now}


def status() -> dict:
    """What the pipeline and panel need to know.

    state: 'unrestricted' (internal build) | 'licensed' | 'trial' |
           'expired'
    """
    if not VENDOR_PUBLIC_KEY:
        return {"state": "unrestricted", "name": "", "edition": "internal",
                "expires": "", "trial_hours_left": 0, "machine": machine_id()}

    payload = decode_key(stored_key())
    if payload is not None:
        bound = payload.get("m")
        expiry = _expiry_date(payload)
        ok_machine = not bound or bound == machine_id()
        ok_date = expiry is None or expiry >= date.today()
        if ok_machine and ok_date:
            return {"state": "licensed",
                    "name": payload.get("n", ""),
                    "edition": payload.get("e", "pro"),
                    "expires": expiry.isoformat() if expiry else "",
                    "trial_hours_left": 0,
                    "machine": machine_id()}

    trial = _trial_state()
    used_h = (trial["now"] - trial["first"]) / 3600.0
    left = max(0.0, TRIAL_HOURS - used_h)
    return {"state": "trial" if left > 0 else "expired",
            "name": "", "edition": "trial", "expires": "",
            "trial_hours_left": round(left, 1),
            "machine": machine_id()}


def is_blocked(st: dict | None = None) -> bool:
    """True when the app must stop producing output."""
    return (st or status())["state"] == "expired"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
