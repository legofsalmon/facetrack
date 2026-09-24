"""Licence keys and the trial clock.

Two kinds of licence unlock a sold build, and either is enough:

* **Shop licences** from letissier.ie (``LT-XXXX-XXXX-XXXX-XXXX``), which
  is what a buyer gets today. Activating one is an online call that takes
  a seat and returns a signed token; from then on the token is checked
  offline and refreshed by a daily check-in. See "shop licences" below.
* **YW1 keys**, minted by the Licence Admin before the shop existed. They
  keep working exactly as they did, so nobody holding one is locked out.

YW1 keys are Ed25519-signed blobs that the app verifies **offline** with an
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
import functools
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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

try:
    from ._buildinfo import BUILD_DATE             # type: ignore
except ImportError:
    BUILD_DATE = 0          # a source run is never "newer than" a licence

TRIAL_HOURS = 72
PRODUCT = "yewee"
_KEY_PREFIX = "YW1."


# ---------------------------------------------------------------- paths

from .paths import user_data_dir  # noqa: E402  (one implementation)


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


def _hardware_id() -> str:
    """The platform's own machine id (macOS IOPlatformUUID, Windows
    MachineGuid), exactly as the OS reports it, or "" if unreadable."""
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
    return raw


def machine_id() -> str:
    """Stable-ish per-machine fingerprint (hashed, never the raw serial).

    This is the id YW1 keys are node-locked to, so it must never change:
    a different answer here strands every machine-bound key already sold.
    """
    raw = _hardware_id()
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


# -------------------------------------------------------- shop licences
#
# What letissier.ie sells: an LT- key, activated online against the
# studio's licence service (https://letissier.ie/integrate). Activation
# takes a seat and returns a token, base64url(claims).base64url(signature),
# signed with the studio's key. The app keeps the token and checks it
# offline from then on; a daily check-in swaps it for a fresh one.
#
# The verification is the SDK's decision (letissier.ie/integrate/sdk/python)
# rewritten over _ed25519 so the app gains no crypto dependency, and made
# stricter in the same two ways as Light's: the product claim must be
# "yewee", and there is one acceptance path for the signature (the ASCII of
# the payload segment, which is what the published vectors sign).

# The studio's signing key, as its integration page publishes it. Compiled
# in rather than read from the environment on purpose: a build that shipped
# without the variable would verify nothing, silently. A wrong key fails
# loudly; a missing one would not.
SHOP_PUBLIC_KEY = "1fca6c21f2eb7963fd646272a731a41a191d3a4cda839e295c5cda67978fcc85"
SHOP_BASE = "https://letissier.ie"
SHOP_ACCOUNT_URL = SHOP_BASE + "/account"
# The shop sells Yewee by major version ("every 1.x build is included"), so
# its update window never runs out and a 2.0 is told apart by product id.
SHOP_PRODUCT = "yewee"

ACTIVE = "active"
UPDATE_REQUIRED = "update_required"
CHECK_IN_REQUIRED = "check_in_required"
EXPIRED = "expired"
WRONG_MACHINE = "wrong_machine"
INVALID = "invalid"

# What each of the service's statuses costs a Yewee operator. A bought
# licence is never taken away by the clock: an ended update window costs
# newer builds, and a lapsed lease only asks for a check-in, because a show
# machine may sit offline for weeks and must still run. Only a trial ends.
_SHOP_UNLOCKS = {ACTIVE, UPDATE_REQUIRED, CHECK_IN_REQUIRED}

# Key groups are 4 characters from this alphabet, the last group a checksum
# over the others; the first group names the product (see the shop's
# src/lib/licensing/licence-key.ts, which this mirrors).
_SHOP_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_SHOP_TAGS = {"YEWE": "Yewee", "11GH": "Light", "V1ZZ": "Vizz",
              "DATA": "Datamosh", "CREW": "Crewbox"}


class ShopError(Exception):
    """The licence service said no, or could not be reached (reason
    'network'). `reason` is the service's own code; see its guide."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def _shop_checksum(body: str) -> str:
    total = 0
    for ch in body.replace("-", ""):
        total = (total * 33 + _SHOP_ALPHABET.index(ch) + 1) % len(_SHOP_ALPHABET) ** 4
    out = ""
    for _ in range(4):
        out = _SHOP_ALPHABET[total % len(_SHOP_ALPHABET)] + out
        total //= len(_SHOP_ALPHABET)
    return out


def normalise_shop_key(text: str) -> str:
    """Canonical LT- form, folding what gets misread down a phone line:
    lower case, spaces, a missing prefix, I/L for 1, O for 0, U for V."""
    raw = "".join((text or "").split()).upper()
    for prefix in ("LT-", "1T-"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    body = raw.replace("I", "1").replace("L", "1").replace("O", "0").replace("U", "V")
    return f"LT-{body}"


def shop_key_problem(key: str) -> str:
    """Why a typed key cannot be a Yewee shop key, or "" if it can.
    Catches typos and wrong-product keys before any network call."""
    parts = normalise_shop_key(key).split("-")
    if len(parts) != 5 or any(len(g) != 4 or any(c not in _SHOP_ALPHABET for c in g)
                              for g in parts[1:]):
        return ("That doesn't look like a licence key. It should read "
                "LT-YEWE- followed by three groups of four.")
    if _shop_checksum("-".join(parts[1:4])) != parts[4]:
        return "That key has a typo in it. Check it against your email."
    if parts[1] != "YEWE":
        other = _SHOP_TAGS.get(parts[1], "another product")
        return f"That key is for {other}, not Yewee."
    return ""


@functools.lru_cache(maxsize=1)
def shop_fingerprint() -> str:
    """The raw machine id the licence service is sent (it hashes it).

    Never hashed here, never logged. The panel shows it only as the request
    code for offline activation, which the account page asks for. Unlike
    machine_id() there is no MAC-address fallback: a MAC changes with a
    dock or a USB adapter, and each change would burn a seat.
    """
    raw = _hardware_id()
    if not raw and sys.platform.startswith("linux"):
        try:
            raw = Path("/etc/machine-id").read_text()
        except OSError:
            raw = ""
    return raw.strip()


def shop_machine_hash(fingerprint: str) -> str:
    """How the service names a machine: sha256 of the trimmed fingerprint,
    first 32 hex characters. For comparing with a token, never the wire."""
    return hashlib.sha256(fingerprint.strip().encode("utf-8")).hexdigest()[:32]


def check_token(token: str, fingerprint: str, build_date: int, now: int,
                public_key_hex: str = SHOP_PUBLIC_KEY,
                product: str = SHOP_PRODUCT) -> tuple[str, dict | None]:
    """The whole offline decision for a shop token: (status, claims).

    Claims come back only once the signature has verified. Order matches
    the SDK and Light: machine, then the lease (`exp`, lapsed at exactly
    exp), then the update window (`maintUntil`, inclusive).
    """
    token = "".join((token or "").split())
    parts = token.split(".")
    if len(parts) != 2 or not all(parts):
        return INVALID, None
    try:
        sig = _b64d(parts[1])
        if not ed.verify(bytes.fromhex(public_key_hex), parts[0].encode("ascii"), sig):
            return INVALID, None
        claims = json.loads(_b64d(parts[0]))
    except Exception:
        return INVALID, None
    if not isinstance(claims, dict) or claims.get("v") != 1:
        return INVALID, None
    if str(claims.get("product", "")).lower() != product:
        return INVALID, None
    try:
        exp, maint = int(claims["exp"]), int(claims["maintUntil"])
    except (KeyError, TypeError, ValueError):
        return INVALID, None
    if str(claims.get("machine", "")).lower() != shop_machine_hash(fingerprint):
        return WRONG_MACHINE, claims
    if now >= exp:
        trial = str(claims.get("edition", "")).lower() == "trial"
        return (EXPIRED if trial else CHECK_IN_REQUIRED), claims
    if build_date > maint:
        return UPDATE_REQUIRED, claims
    return ACTIVE, claims


def _shop_path() -> Path:
    return user_data_dir() / "shop-licence.json"


def _read_shop() -> dict:
    try:
        rec = json.loads(_shop_path().read_text())
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _write_shop(rec: dict) -> None:
    path = _shop_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, path)        # never a half-written licence


def _ssl_context():
    import ssl
    # A packaged Python on macOS has no system CA bundle of its own, so
    # certifi's is used when it was bundled; plain urllib otherwise.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _post(path: str, body: dict, timeout: float = 15.0) -> dict:
    from . import app_version
    req = urllib.request.Request(
        SHOP_BASE + path, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": f"yewee/{app_version()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=_ssl_context()) as resp:
            reply = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            failed = json.loads(exc.read() or b"{}")
        except ValueError:
            failed = {}
        if not isinstance(failed, dict):
            failed = {}
        raise ShopError(failed.get("reason") or f"http_{exc.code}",
                        failed.get("message")
                        or f"letissier.ie refused the request ({exc.code}).") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ShopError("network", "Couldn't reach letissier.ie.") from exc
    if not isinstance(reply, dict) or reply.get("ok") is False:
        raise ShopError("bad_reply", "letissier.ie sent back something unreadable.")
    return reply


def _accept_token(token: str, key: str, fingerprint: str,
                  now: int | None = None) -> dict:
    """Store a token only if it unlocks this machine, and say why not."""
    moment = int(time.time()) if now is None else now
    st, claims = check_token(token, fingerprint, BUILD_DATE, moment, SHOP_PUBLIC_KEY)
    if st == WRONG_MACHINE:
        raise ShopError("wrong_machine",
                        "That licence was issued for a different machine. Nothing "
                        "was stored. For offline activation, the request code has "
                        f"to be typed exactly as shown: {fingerprint}")
    if st == INVALID or claims is None:
        raise ShopError("invalid", "That isn't a valid Yewee licence.")
    if st == EXPIRED:
        raise ShopError("expired", "That trial has already ended.")
    _write_shop({"key": key or str(claims.get("key", "")), "token": token})
    return claims


def _thanks(claims: dict) -> str:
    return f"Activated. Thank you, {claims.get('name') or 'friend'}."


def activate_shop_key(key: str) -> tuple[bool, str]:
    """Online activation of an LT- key: takes one of the licence's seats."""
    problem = shop_key_problem(key)
    if problem:
        return False, problem
    key = normalise_shop_key(key)
    fingerprint = shop_fingerprint()
    if not fingerprint:
        return False, ("Can't read this machine's hardware id, so it can't "
                       "take a licence seat. Please tell us at info@letissier.ie.")
    try:
        # "product" makes the service refuse another product's key before
        # it takes a seat (reason wrong_product). shop_key_problem() has
        # already caught that from the key's first group; this is the
        # server's word on it, for keys whose group and product disagree.
        reply = _post("/api/licence/activate",
                      {"key": key, "machine": fingerprint, "product": SHOP_PRODUCT,
                       "label": f"Yewee on {socket.gethostname()}"})
        # The service echoes the hash it recorded. If that is not ours the
        # token can never verify here, and re-activating would only repeat it.
        if reply.get("machine") not in (None, shop_machine_hash(fingerprint)):
            raise ShopError("machine_mismatch",
                            "letissier.ie recorded a different machine id than this "
                            "machine has. Nothing was stored.")
        claims = _accept_token(str(reply.get("token", "")), key, fingerprint)
    except ShopError as exc:
        if exc.reason == "network":
            return False, ("Couldn't reach letissier.ie to activate. If this machine "
                           "stays offline, sign in at letissier.ie/account on any "
                           "device, choose offline activation, enter this request "
                           f"code, and paste the licence it gives you here: "
                           f"{fingerprint}")
        return False, exc.message
    return True, _thanks(claims)


def activate_shop_token(token: str) -> tuple[bool, str]:
    """Offline activation: a token pasted from the account page."""
    fingerprint = shop_fingerprint()
    try:
        claims = _accept_token("".join(token.split()), "", fingerprint)
    except ShopError as exc:
        return False, exc.message
    return True, _thanks(claims)


def check_in() -> str:
    """One check-in with the service. Returns what happened, for the log.

    A network failure changes nothing: the stored token stays the answer.
    Only the service saying the licence is over on this machine (revoked,
    as after a refund, or this machine's seat released from the account
    page) ends it, and even then not mid-run; see _shop_status.
    """
    rec = _read_shop()
    key, fingerprint = rec.get("key"), shop_fingerprint()
    if not key or not fingerprint or rec.get("ended"):
        return "nothing to check in"
    try:
        reply = _post("/api/licence/heartbeat", {"key": key, "machine": fingerprint})
        _accept_token(str(reply.get("token", "")), key, fingerprint)
    except ShopError as exc:
        if exc.reason in ("revoked", "not_activated"):
            rec["ended"] = exc.reason
            _write_shop(rec)
            return f"licence ended by letissier.ie ({exc.reason})"
        return f"check-in failed, keeping the stored licence: {exc.message}"
    return "checked in"


_heartbeat_started = False


def start_check_ins(first_after: float = 60.0, every: float = 24 * 3600.0) -> None:
    """Check in a minute after launch and then daily, off the startup path,
    on a daemon thread. Does nothing in an unrestricted build."""
    global _heartbeat_started
    if _heartbeat_started or not VENDOR_PUBLIC_KEY:
        return
    _heartbeat_started = True

    def loop():
        time.sleep(first_after)
        while True:
            try:
                note = check_in()
            except Exception as exc:          # never let licensing break the show
                note = f"check-in error: {exc}"
            if note != "nothing to check in":
                print(f"[yewee] licence: {note}", flush=True)
            time.sleep(every)

    threading.Thread(target=loop, daemon=True, name="yewee-licence").start()


# Set once a shop licence has unlocked this run. The service ending a
# licence (a refund) is honoured at the next launch, never mid-show.
_held_this_run = False


def _shop_status(now: int | None = None) -> tuple[dict | None, str]:
    """(status dict if a shop licence unlocks this machine, else None,
    a note for the panel either way)."""
    global _held_this_run
    rec = _read_shop()
    if not rec.get("token"):
        return None, ""
    moment = int(time.time()) if now is None else now
    fingerprint = shop_fingerprint()
    st, claims = check_token(rec["token"], fingerprint, BUILD_DATE, moment,
                             SHOP_PUBLIC_KEY)
    ended = rec.get("ended")
    if ended and not _held_this_run:
        why = ("it was revoked, usually after a refund" if ended == "revoked"
               else "this machine's seat was released")
        return None, (f"letissier.ie says the licence stored here has ended: {why}. "
                      "Enter a key to activate again.")
    if st not in _SHOP_UNLOCKS or claims is None:
        return None, {
            EXPIRED: "Your letissier.ie trial has ended.",
            WRONG_MACHINE: ("The stored licence belongs to another machine. Activate "
                            "your key here, or release the other seat at "
                            "letissier.ie/account."),
        }.get(st, "The stored licence could not be verified.")
    _held_this_run = True
    note = ""
    if ended:
        note = ("letissier.ie says this licence has ended. Yewee keeps running "
                "until it is quit.")
    elif st == UPDATE_REQUIRED:
        until = datetime.fromtimestamp(int(claims["maintUntil"]), timezone.utc)
        note = (f"Your updates ran to {until.date().isoformat()}. This build is newer; "
                "it keeps running, and renewing at letissier.ie/account covers it.")
    elif st == CHECK_IN_REQUIRED:
        note = ("Not checked in for a while. Nothing is restricted; connect this "
                "machine to the internet once to refresh the licence.")
    trial = str(claims.get("edition", "")).lower() == "trial"
    left = max(0.0, (int(claims["exp"]) - moment) / 3600.0)
    return {"state": "trial" if trial else "licensed",
            "source": "shop",
            "name": claims.get("name") or "",
            "edition": str(claims.get("edition") or "standard"),
            "expires": "",
            "trial_hours_left": round(left, 1) if trial else 0,
            "key": rec.get("key", ""),
            "note": note}, note


# ------------------------------------------------------------- storage

def _licence_path() -> Path:
    return user_data_dir() / "licence.key"


def stored_key() -> str:
    try:
        return _licence_path().read_text().strip()
    except OSError:
        return ""


def activate(key: str) -> tuple[bool, str]:
    """Verify and store whatever the operator pasted: a YW1 key, a shop
    key (activated online), or a shop licence from offline activation.
    Returns (ok, message for the operator). May block on the network for
    a shop key, so call it off the event loop."""
    if not VENDOR_PUBLIC_KEY:
        return False, "This build does not use licence keys."
    key = "".join((key or "").split())
    if not key:
        return False, "Paste a licence key first."
    if not key.startswith(_KEY_PREFIX):
        if key.count(".") == 1 and len(key) > 64:
            return activate_shop_token(key)
        return activate_shop_key(key)
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


def deactivate() -> str:
    """Forget every licence on this machine. A shop licence's seat is
    released too when the service can be reached; a machine with no
    network must still be able to forget its licence, so that part is
    best effort. Returns a message for the operator."""
    global _held_this_run
    rec = _read_shop()
    released = False
    if rec.get("key") and not rec.get("ended") and shop_fingerprint():
        try:
            _post("/api/licence/deactivate",
                  {"key": rec["key"], "machine": shop_fingerprint()}, timeout=8.0)
            released = True
        except ShopError:
            pass
    for path in (_licence_path(), _shop_path()):
        try:
            path.unlink()
        except OSError:
            pass
    _held_this_run = False
    if rec.get("key") and not released and not rec.get("ended"):
        return ("Licence removed from this machine. Its seat couldn't be released "
                "from here; do that at letissier.ie/account.")
    if released:
        return "Licence removed from this machine, and its seat released."
    return "Licence removed from this machine."


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

    A shop licence is looked at first, then a YW1 key, then the trial
    clock. `note` carries anything the operator should read (a lapsed
    check-in, an ended licence) without it restricting anything.
    """
    if not VENDOR_PUBLIC_KEY:
        return {"state": "unrestricted", "name": "", "edition": "internal",
                "expires": "", "trial_hours_left": 0, "machine": machine_id()}

    shop, note = _shop_status()
    if shop is not None:
        shop["machine"] = machine_id()
        shop["request_code"] = shop_fingerprint()
        return shop

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
                    "machine": machine_id(),
                    "request_code": shop_fingerprint(),
                    "source": "yw1", "note": ""}

    trial = _trial_state()
    used_h = (trial["now"] - trial["first"]) / 3600.0
    left = max(0.0, TRIAL_HOURS - used_h)
    return {"state": "trial" if left > 0 else "expired",
            "name": "", "edition": "trial", "expires": "",
            "trial_hours_left": round(left, 1),
            "machine": machine_id(),
            "request_code": shop_fingerprint(),
            "source": "", "note": note}


def is_blocked(st: dict | None = None) -> bool:
    """True when the app must stop producing output."""
    return (st or status())["state"] == "expired"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
