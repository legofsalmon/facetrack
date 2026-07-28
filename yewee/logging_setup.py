"""Mirror console output to a rotating log file (logs/yewee.log).

Everything printed to stdout/stderr — including tracebacks — also lands
in the log, so there's a record to check after an unattended crash.
uvicorn/library loggers are routed to the same file.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class _Tee:
    def __init__(self, stream, logger: logging.Logger, level: int):
        self._stream = stream
        self._logger = logger
        self._level = level
        self._buf = ""

    def write(self, data: str) -> int:
        n = self._stream.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line)
        return n

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()


def setup(root_dir: str | None = None) -> str:
    """Install the tee; returns the log file path (best-effort — console
    behaviour is unchanged if the log directory can't be created).

    root_dir is ignored and kept for callers that still pass it; the
    location comes from paths.log_dir() so a packaged app never writes
    inside its own bundle."""
    from .paths import log_dir as _log_dir, log_path as _log_path
    log_dir = str(_log_dir())
    log_path = str(_log_path())
    try:
        os.makedirs(log_dir, exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=2_000_000,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        console = logging.getLogger("yewee.console")
        console.setLevel(logging.INFO)
        console.addHandler(handler)
        console.propagate = False
        sys.stdout = _Tee(sys.stdout, console, logging.INFO)
        sys.stderr = _Tee(sys.stderr, console, logging.ERROR)
        root = logging.getLogger()
        root.addHandler(handler)  # uvicorn etc.
        if root.level > logging.WARNING or root.level == logging.NOTSET:
            root.setLevel(logging.WARNING)
    except OSError:
        pass
    return log_path
