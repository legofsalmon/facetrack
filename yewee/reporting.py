"""Crash reports and feedback to LeTissier Creative Studios — only with consent.

The contract is letissier.ie's intake API (POST /api/reports/crash and
/api/reports/feedback); the rules this module keeps are the contract's:

1. **Opt-in.** "Send crash reports automatically" is off until the operator
   turns it on. After an unclean exit the panel asks once; without the
   setting or that click nothing is sent. Feedback goes only when someone
   presses Send in the feedback form.
2. **Scrubbed here, before anything is stored for sending**: the home
   directory becomes ~, user names drop out of paths, URLs lose their
   query strings, and licence keys, e-mail addresses, IP addresses, this
   machine's name and the names of its sources and feeds are replaced.
3. **Offline first.** Consented reports wait as JSON files in
   <data dir>/reports/queue (at most 20, oldest dropped) and go when there
   is a network: a background thread a few seconds after launch, and right
   after something new is queued. Never on the frame loop, never blocking
   startup, 8 s timeout, and a failure is never shown as an error.

A crash report never carries a licence key, an e-mail address, a name, a
document or source path, another device's IP address, NDI/Syphon/Spout
source names, settings contents, camera frames or audio. The optional note
the operator types is stored by the studio and never posted publicly.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import app_version, settings
from .paths import state_dir

PRODUCT = "yewee"
#: The studio's intake. LETISSIER_API overrides it for tests only.
API_BASE = (os.environ.get("LETISSIER_API") or "https://letissier.ie").rstrip("/")
TIMEOUT = 8.0
MAX_QUEUE = 20
MAX_BODY = 64 * 1024

KINDS = ("panic", "exception", "signal", "unclean-exit", "gpu", "hang", "other")
FEEDBACK_TYPES = ("bug", "idea", "question", "praise")
LIMITS = {"version": 32, "osVersion": 32, "arch": 16, "install": 64,
          "summary": 300, "detail": 32768, "signature": 128, "note": 2000,
          "message": 5000, "email": 254, "name": 100}
CRASH_FIELDS = ("product", "version", "os", "osVersion", "arch", "install", "kind",
                "summary", "detail", "signature", "occurredAt", "note")
FEEDBACK_FIELDS = ("product", "version", "os", "osVersion", "arch", "install", "type",
                   "message", "email", "name", "licence", "public")
ENDPOINTS = {"crash": "/api/reports/crash", "feedback": "/api/reports/feedback"}

_lock = threading.RLock()
_flush_lock = threading.Lock()
_state: dict | None = None          # the "reports" section of settings.json
_prompt: dict | None = None         # crash report waiting for the operator
_secrets: set[str] = set()          # strings that must never leave the machine
_nonfatal_seen: set[str] = set()


# ------------------------------------------------------------- consent

def _section() -> dict:
    global _state
    with _lock:
        if _state is None:
            _state = settings.load_section("reports")
        return _state


def _save() -> None:
    with _lock:
        settings.save_section("reports", dict(_section()))


def install_id() -> str:
    """A random id made once per install and kept with the settings. Only
    counts affected installs; never derived from the machine, the licence
    or the person."""
    with _lock:
        state = _section()
        iid = str(state.get("install") or "")
        if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", iid):
            iid = str(uuid.uuid4())
            state["install"] = iid
            _save()
        return iid


def auto_send() -> bool:
    return bool(_section().get("auto_send", False))


def set_auto_send(on: bool) -> None:
    with _lock:
        _section()["auto_send"] = bool(on)
        _save()


# ------------------------------------------------------------ platform

def os_name() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def os_version() -> str:
    try:
        if sys.platform == "darwin":
            return platform.mac_ver()[0][:32]
        if sys.platform == "win32":
            return platform.version()[:32]
        return platform.release()[:32]
    except Exception:
        return ""


def arch() -> str:
    machine = (platform.machine() or "").lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64",
            "arm64": "arm64", "x86_64": "x86_64"}.get(machine, machine)[:16]


# ------------------------------------------------------------ scrubbing

def add_secrets(*values) -> None:
    """Strings this run knows are private (the machine's name, the user
    name, source specs and feed names) and that the scrubber replaces."""
    with _lock:
        for value in values:
            text = str(value or "").strip()
            for prefix in ("cam:", "ndi:"):
                if text.lower().startswith(prefix):
                    text = text[len(prefix):].strip()
            if len(text) >= 3 and not text.isdigit():
                _secrets.add(text)


_USER_PATH = re.compile(r"(?i)(/Users/|/home/|[A-Z]:[\\/]+(?:Users|Documents and Settings)[\\/]+)"
                        r"[^/\\\s\"'<>:]+")
_URL_QUERY = re.compile(r"((?:https?|wss?|ftp|rtsp|rtmp|srt|udp)://[^\s?#\"'<>]*)[?#][^\s\"'<>]*")
_URL_AUTH = re.compile(r"((?:https?|wss?|ftp|rtsp|rtmp|srt)://)[^\s/@\"'<>]+@")
_LICENCE = re.compile(r"\b(?:LT|1T)-[0-9A-Z]{4}(?:-[0-9A-Z]{4}){2,4}\b|YW1\.[\w-]+\.[\w-]+",
                      re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                   r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}(?![\w:])", re.I)
# lists of what is on the network or plugged in, as yewee's own errors
# print them ("Visible sources: [...]", "connected now: [...]")
_LISTED = re.compile(r"(?i)((?:visible sources|connected now):\s*)(\[[^\]]*\]|'[^']*'|\S+)")


def scrub(text: str, extra=()) -> str:
    """Remove what must not leave the machine from free text (see the
    module docstring). Order matters: specific secrets before patterns."""
    if not text:
        return ""
    out = str(text)
    homes = set()
    try:
        home = str(Path.home())
        homes |= {home, home.replace("\\", "/"), home.replace("/", "\\")}
    except Exception:
        pass
    for h in sorted((h for h in homes if len(h) > 3), key=len, reverse=True):
        out = re.sub(re.escape(h), "~", out, flags=re.I if sys.platform == "win32" else 0)
    out = _USER_PATH.sub(lambda m: m.group(1) + "<user>", out)
    out = _URL_AUTH.sub(r"\1", out)
    out = _URL_QUERY.sub(r"\1", out)
    out = _LISTED.sub(r"\1<redacted>", out)
    out = _LICENCE.sub("<licence>", out)
    out = _EMAIL.sub("<email>", out)
    with _lock:
        private = set(_secrets)
    try:
        import getpass
        private.add(getpass.getuser())
    except Exception:
        pass
    try:
        import socket
        private.add(socket.gethostname())
        private.add(socket.gethostname().split(".")[0])
    except Exception:
        pass
    for s in extra:
        if s and len(str(s)) >= 3:
            private.add(str(s))
    for s in sorted((s for s in private if s and len(s) >= 3), key=len, reverse=True):
        # whole words only: a machine called "Mac" must not eat "macos"
        out = re.sub(r"(?<!\w)" + re.escape(s) + r"(?!\w)", "<redacted>", out,
                     flags=re.I)
    out = _IPV4.sub(lambda m: m.group(0) if m.group(0).startswith("127.") else "<ip>", out)
    out = _IPV6.sub("<ip>", out)
    return out


def _cut(text: str, limit: int, keep_tail: bool = False) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    if keep_tail:
        head = limit // 4
        return text[:head] + "\n…\n" + text[-(limit - head - 3):]
    return text[:limit - 1] + "…"


# ------------------------------------------------------------- payloads

_FRAME = re.compile(r'File "([^"]+)", line \d+,? in ([^\s]+)')


def signature(kind: str, summary: str, detail: str) -> str:
    """The same crash, the same id: kind, exception type and the innermost
    frames (file basenames and function names — no line numbers, paths or
    addresses, so it survives a rebuild and differs between machines not at
    all)."""
    etype = (summary or "").split(":", 1)[0].strip()[:80]
    frames = [f"{os.path.basename(f.replace(chr(92), '/'))}:{fn}"
              for f, fn in _FRAME.findall(detail or "")][-6:]
    raw = "|".join([kind, etype] + frames)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:12]


def _base() -> dict:
    return {"product": PRODUCT, "version": _cut(app_version(), LIMITS["version"]),
            "os": os_name(), "osVersion": os_version(), "arch": arch(),
            "install": install_id()}


def crash_payload(kind: str, summary: str, detail: str = "", note: str = "",
                  occurred_at: str | None = None, extra_secrets=()) -> dict:
    """The body of one crash report: scrubbed, cut to the contract's limits,
    and holding the contract's fields only."""
    kind = kind if kind in KINDS else "other"
    summary = scrub(summary, extra_secrets).strip().splitlines()
    summary = _cut(summary[-1] if summary else "(no message)", LIMITS["summary"])
    detail = _cut(scrub(detail, extra_secrets), LIMITS["detail"], keep_tail=True)
    body = _base()
    body.update({"kind": kind, "summary": summary,
                 "signature": signature(kind, summary, detail)})
    if detail:
        body["detail"] = detail
    if occurred_at:
        body["occurredAt"] = str(occurred_at)[:40]
    note = _cut(scrub(note, extra_secrets).strip(), LIMITS["note"])
    if note:
        body["note"] = note
    while len(json.dumps(body).encode("utf-8")) > MAX_BODY and body.get("detail"):
        body["detail"] = _cut(body["detail"], len(body["detail"]) // 2, keep_tail=True)
    return {k: body[k] for k in CRASH_FIELDS if k in body}


def feedback_payload(kind: str, message: str, email: str = "", name: str = "",
                     licence: str = "", public: bool = False) -> dict:
    """The body of one piece of feedback. Raises ValueError with a message
    for the person when it cannot be sent as typed. Nothing here is
    scrubbed: the person wrote it to be read, and sees it before sending."""
    if kind not in FEEDBACK_TYPES:
        raise ValueError("Choose what kind of feedback this is.")
    message = (message or "").strip()
    if not message:
        raise ValueError("Write a message first.")
    if len(message) > LIMITS["message"]:
        raise ValueError(f"That is {len(message)} characters; the limit is "
                         f"{LIMITS['message']}.")
    email = (email or "").strip()
    if email and (len(email) > LIMITS["email"] or not _EMAIL.fullmatch(email)):
        raise ValueError("That e-mail address doesn't look right.")
    body = _base()
    body.update({"type": kind, "message": message, "public": bool(public)})
    if email:
        body["email"] = email
    name = (name or "").strip()[:LIMITS["name"]]
    if name:
        body["name"] = name
    if licence:
        body["licence"] = str(licence).strip()
    return {k: body[k] for k in FEEDBACK_FIELDS if k in body}


# ---------------------------------------------------------------- queue

def queue_dir() -> Path:
    d = state_dir() / "reports" / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def queued() -> list[Path]:
    try:
        return sorted(queue_dir().glob("*.json"))
    except OSError:
        return []


def enqueue(endpoint: str, payload: dict) -> Path | None:
    """Keep a consented report until it can be delivered."""
    if endpoint not in ENDPOINTS:
        raise ValueError(endpoint)
    d = queue_dir()
    path = d / f"{time.time_ns()}-{endpoint}.json"
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps({"endpoint": endpoint, "payload": payload}),
                       encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return None
    items = queued()
    for old in items[:max(0, len(items) - MAX_QUEUE)]:
        try:
            old.unlink()
        except OSError:
            pass
    return path


def _http_post(endpoint: str, payload: dict) -> int:
    """POST one report; returns the HTTP status. Raises OSError when the
    service cannot be reached at all."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_BASE + ENDPOINTS[endpoint], data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": f"Yewee/{app_version()} ({os_name()})"})
    context = None
    if API_BASE.startswith("https:"):
        from .licensing import _ssl_context        # certifi when bundled
        context = _ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=context) as resp:
            resp.read(4096)
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, ValueError) as exc:
        raise OSError(str(exc)) from exc


#: Replaced in tests. Returns an HTTP status or raises OSError.
_send = _http_post


def flush() -> dict:
    """Deliver what is queued, oldest first. 2xx and 400/413 (the service
    will never take it) remove an item; 429, any other status or no network
    keeps it and stops until next time."""
    counts = {"sent": 0, "dropped": 0, "kept": 0}
    if not _flush_lock.acquire(timeout=TIMEOUT * 2):
        return counts
    try:
        items = queued()
        for i, path in enumerate(items):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                endpoint, payload = item["endpoint"], item["payload"]
                if endpoint not in ENDPOINTS or not isinstance(payload, dict):
                    raise ValueError
            except (OSError, ValueError, KeyError, TypeError):
                _drop(path)
                counts["dropped"] += 1
                continue
            try:
                status = _send(endpoint, payload)
            except OSError:
                counts["kept"] += len(items) - i
                break
            if 200 <= status < 300:
                _drop(path)
                counts["sent"] += 1
            elif status in (400, 413):
                _drop(path)
                counts["dropped"] += 1
            else:
                counts["kept"] += len(items) - i
                break
    finally:
        _flush_lock.release()
    return counts


def _drop(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def flush_in_background(delay: float = 0.0) -> threading.Thread:
    def run():
        if delay:
            time.sleep(delay)
        try:
            flush()
        except Exception:
            pass                 # never an error anyone sees
    t = threading.Thread(target=run, daemon=True, name="yewee-reports")
    t.start()
    return t


# ------------------------------------------------------ after a crash

def _prompt_path() -> Path:
    d = state_dir() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d / "pending-crash.json"


def handle_previous_session(report: dict | None) -> str:
    """Decide what happens to the report on a run that ended badly:
    'queued' when the operator has said to send automatically, 'prompt'
    when it waits for them to say yes or no in the panel, 'none' when there
    is nothing. Nothing is sent from here without consent."""
    global _prompt
    if report:
        payload = crash_payload(report.get("kind", "other"), report.get("summary", ""),
                                report.get("detail", ""),
                                occurred_at=report.get("occurredAt"))
        if report.get("version"):
            payload["version"] = _cut(str(report["version"]), LIMITS["version"])
        if auto_send():
            enqueue("crash", payload)
            return "queued"
        try:
            _prompt_path().write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
        with _lock:
            _prompt = payload
        return "prompt"
    # an earlier crash the operator has not answered yet keeps its question
    pending = pending_prompt()
    return "prompt" if pending else "none"


def pending_prompt() -> dict | None:
    global _prompt
    with _lock:
        if _prompt is None:
            try:
                data = json.loads(_prompt_path().read_text(encoding="utf-8"))
                _prompt = data if isinstance(data, dict) else None
            except (OSError, ValueError):
                _prompt = None
        return _prompt


def answer_prompt(send: bool, always: bool = False, note: str = "") -> str:
    """The operator's answer to "closed unexpectedly — send a report?"."""
    global _prompt
    pending = pending_prompt()
    with _lock:
        _prompt = None
    try:
        _prompt_path().unlink()
    except OSError:
        pass
    if always:
        set_auto_send(True)
    if not send or not pending:
        return "Not sent." if not send else "There was no report to send."
    payload = dict(pending)
    note = _cut(scrub(note).strip(), LIMITS["note"])
    if note:
        payload["note"] = note
    enqueue("crash", {k: payload[k] for k in CRASH_FIELDS if k in payload})
    result = flush()
    if result["sent"]:
        return "Sent. Thank you — it helps."
    return "Saved; it will be sent the next time this machine is online."


def note_nonfatal(exc: BaseException) -> bool:
    """An error yewee recovered from (a skipped frame, a dead helper
    thread). Reported only when sending is switched on, once per distinct
    error per run. Returns whether it was queued."""
    import traceback
    if not auto_send():
        return False
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    summary = summary.splitlines()[-1] if summary else type(exc).__name__
    payload = crash_payload("exception", f"{summary} (recovered, the app kept running)",
                            detail, occurred_at=_utc())
    with _lock:
        if payload["signature"] in _nonfatal_seen:
            return False
        _nonfatal_seen.add(payload["signature"])
    enqueue("crash", payload)
    flush_in_background()
    return True


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


# ------------------------------------------------------------- feedback

def submit_feedback(kind: str, message: str, email: str = "", licence: str = "",
                    public: bool = False) -> tuple[bool, str]:
    """Queue feedback the person pressed Send on, and try to deliver it now.
    `licence` is passed only when they ticked "include my licence"."""
    try:
        payload = feedback_payload(kind, message, email=email, licence=licence,
                                   public=public)
    except ValueError as exc:
        return False, str(exc)
    if enqueue("feedback", payload) is None:
        return False, "Couldn't save it on this machine. Please try again."
    result = flush()
    if result["sent"]:
        return True, "Sent. Thank you."
    return True, "Saved on this machine; it will be sent when it is next online."


# ---------------------------------------------------------------- panel

def panel_state() -> dict:
    """What the panel shows: the setting, any question waiting, the queue."""
    pending = pending_prompt()
    return {"auto_send": auto_send(),
            "prompt": ({"kind": pending.get("kind"), "summary": pending.get("summary"),
                        "payload": pending} if pending else None),
            "queued": len(queued())}


def _reset_for_tests() -> None:
    global _state, _prompt
    with _lock:
        _state = None
        _prompt = None
        _secrets.clear()
        _nonfatal_seen.clear()
