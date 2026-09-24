"""Knowing, on the next launch, that the last run ended badly — and why.

Nothing here talks to the network: it only leaves evidence on this
machine. yewee/reporting.py decides what, if anything, leaves it.

How a run is followed, per process (several instances can share a data
directory, so every file carries the pid):

* ``crash/running-<pid>.json`` — the crash marker, written when the app
  starts and removed on every clean exit (Quit, Ctrl-C, SIGTERM/SIGHUP,
  the panel's Restart, closing the console window). Finding one whose
  process is gone means that run ended uncleanly.
* ``crash/fault-<pid>.log`` — faulthandler writes the Python stack of every
  thread here if the process dies of a signal (a segfault inside OpenCV,
  ONNX Runtime or the NDI library) — the case no Python hook can see.
* ``crash/last-<pid>.json`` — the uncaught exception, from sys.excepthook
  (main thread) or threading.excepthook (any other).
* ``crash/hang-<pid>.log`` — every thread's stack, written by the watchdog
  just before it kills a pipeline that stopped responding.

On the next launch begin() turns whichever of those exists into one report
dict (kind, summary, detail, occurredAt, version) and clears them. The
settings the operator had (source, feeds, every panel value) are restored
by settings.load() as on any launch; this module only explains the crash.
"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .paths import state_dir

_lock = threading.Lock()
_fault_file = None
_active = False
_prev_excepthook = None
_prev_threading_hook = None
_console_handler = None        # keeps the Windows ctypes callback alive
_dir: Path | None = None       # where this run's files live (set by begin)

#: Called with the exception when a non-main thread dies (the app carries
#: on without it). reporting.note_nonfatal is wired in by main.
on_thread_exception = None


def crash_dir() -> Path:
    d = state_dir() / "crash"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _where(d: Path | None) -> Path:
    return d or _dir or crash_dir()


def _utc(ts: float | None = None) -> str:
    moment = datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _read_text(path: Path, limit: int = 256 * 1024) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """Whether a process with this pid is running now. Never signals it:
    on Windows os.kill(pid, 0) would terminate the process."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
                return bool(ok) and code.value == 259      # STILL_ACTIVE
            finally:
                k32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _files(d: Path, pid: int) -> dict[str, Path]:
    return {"marker": d / f"running-{pid}.json", "fault": d / f"fault-{pid}.log",
            "last": d / f"last-{pid}.json", "hang": d / f"hang-{pid}.log"}


def _first_line(text: str, prefer: tuple[str, ...] = ()) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for want in prefer:
        for ln in lines:
            if ln.startswith(want):
                return ln
    return lines[0] if lines else ""


def _report_for(files: dict[str, Path], marker: dict) -> dict:
    """What is known about one run that did not end cleanly."""
    version = str(marker.get("version") or "")
    last = _read_json(files["last"])
    if last and last.get("summary"):
        return {"kind": last.get("kind") or "exception", "summary": last["summary"],
                "detail": last.get("detail", ""), "occurredAt": last.get("at"),
                "version": version}
    hang = _read_text(files["hang"])
    if hang.strip():
        return {"kind": "hang",
                "summary": _first_line(hang) or "The pipeline stopped responding",
                "detail": hang, "occurredAt": _mtime(files["hang"]), "version": version}
    fault = _read_text(files["fault"])
    if fault.strip():
        return {"kind": "signal",
                "summary": _first_line(fault, ("Fatal Python error",
                                               "Windows fatal exception")),
                "detail": fault, "occurredAt": _mtime(files["fault"]), "version": version}
    return {"kind": "unclean-exit",
            "summary": "Yewee closed unexpectedly, and nothing was recorded about why "
                       "(power loss, a forced quit, or the process being killed)",
            "detail": "", "occurredAt": None, "version": version}


def _mtime(path: Path) -> str | None:
    try:
        return _utc(path.stat().st_mtime)
    except OSError:
        return None


def previous_session(d: Path | None = None) -> dict | None:
    """The report for the most recent run that ended uncleanly, or None.
    Clears the evidence of every finished run it looked at; runs still
    alive (another instance) are left alone."""
    d = d or crash_dir()
    found: list[tuple[float, dict]] = []
    for marker_path in d.glob("running-*.json"):
        try:
            pid = int(marker_path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            _unlink(marker_path)
            continue
        if pid_alive(pid) and pid != os.getpid():
            continue                      # another instance, still running
        marker = _read_json(marker_path) or {}
        files = _files(d, pid)
        report = _report_for(files, marker)
        try:
            started = float(marker.get("t", 0))
        except (TypeError, ValueError):
            started = 0.0
        found.append((started, report))
        for path in files.values():
            _unlink(path)
    # A last-*.json with no marker belongs to a run that exited cleanly
    # after a thread died; nothing to report, so just tidy it away.
    for stray in list(d.glob("last-*.json")) + list(d.glob("hang-*.log")):
        pid = stray.stem.split("-", 1)[-1]
        if not (d / f"running-{pid}.json").exists():
            _unlink(stray)
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[-1][1]


def begin(version: str, d: Path | None = None) -> dict | None:
    """Start following this run. Returns the report on the previous run if
    it ended uncleanly (see previous_session), else None."""
    global _fault_file, _active, _dir
    d = d or crash_dir()
    _dir = d
    previous = previous_session(d)
    files = _files(d, os.getpid())
    _write_json(files["marker"], {"pid": os.getpid(), "version": version,
                                  "started": _utc(), "t": time.time()})
    try:
        # must stay open for the life of the process: faulthandler writes
        # to its file descriptor from a signal handler
        _fault_file = open(files["fault"], "w", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except (OSError, RuntimeError, ValueError):
        _fault_file = None
    _install_hooks()
    _active = True
    return previous


def end_clean(d: Path | None = None) -> None:
    """This run is ending on purpose: forget the marker and the evidence."""
    global _fault_file, _active
    with _lock:
        if not _active:
            return
        _active = False
        try:
            faulthandler.disable()
        except Exception:
            pass
        if _fault_file is not None:
            try:
                _fault_file.close()
            except OSError:
                pass
            _fault_file = None
        for path in _files(_where(d), os.getpid()).values():
            _unlink(path)


def is_active() -> bool:
    return _active


def exception_record(exc_type, exc, tb, kind: str = "exception",
                     thread: str = "") -> dict:
    """The local record of one exception: summary line and full traceback."""
    summary = "".join(traceback.format_exception_only(exc_type, exc)).strip()
    summary = summary.splitlines()[-1] if summary else exc_type.__name__
    if thread:
        summary = f"in thread {thread}: {summary}"
    detail = "".join(traceback.format_exception(exc_type, exc, tb))
    return {"kind": kind, "summary": summary, "detail": detail, "at": _utc()}


def record_exception(exc_type, exc, tb, kind: str = "exception", thread: str = "",
                     d: Path | None = None) -> None:
    """Keep this exception on disk in case the process does not survive it."""
    try:
        _write_json(_files(_where(d), os.getpid())["last"],
                    exception_record(exc_type, exc, tb, kind, thread))
    except Exception:
        pass


def record_hang(reason: str, d: Path | None = None) -> None:
    """Called by the watchdog just before it kills a stalled process: the
    stack of every thread says where it was stuck."""
    try:
        with open(_files(_where(d), os.getpid())["hang"], "w",
                  encoding="utf-8") as f:
            f.write(reason.strip() + "\n\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
    except Exception:
        pass


def _install_hooks() -> None:
    global _prev_excepthook, _prev_threading_hook
    if _prev_excepthook is not None:
        return
    _prev_excepthook = sys.excepthook
    _prev_threading_hook = threading.excepthook

    def excepthook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            record_exception(exc_type, exc, tb)
        _prev_excepthook(exc_type, exc, tb)

    def thread_hook(args):
        if args.exc_type is not SystemExit:
            name = getattr(args.thread, "name", "") or "thread"
            record_exception(args.exc_type, args.exc_value, args.exc_traceback,
                             thread=name)
            if on_thread_exception is not None and args.exc_value is not None:
                try:
                    on_thread_exception(args.exc_value)
                except Exception:
                    pass
        _prev_threading_hook(args)

    sys.excepthook = excepthook
    threading.excepthook = thread_hook


def watch_console_close(stop) -> None:
    """Windows: closing the console window, logging off or shutting down
    kills the process a few seconds later with no signal Python sees. Treat
    it as the deliberate quit it is — stop the pipeline, give main a moment
    to finish, and clear the marker — so it is not reported as a crash."""
    global _console_handler
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def handler(event):
            if event in (2, 5, 6):    # CLOSE, LOGOFF, SHUTDOWN
                try:
                    stop()
                except Exception:
                    pass
                deadline = time.monotonic() + 3.0
                while _active and time.monotonic() < deadline:
                    time.sleep(0.05)
                end_clean()
            return False              # let Windows (or Python) carry on

        _console_handler = handler_type(handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
    except Exception:
        pass
