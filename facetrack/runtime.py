"""CPU budget for the ML runtimes.

OpenCV and ONNX Runtime both default to "use every core", so a busy
pipeline can saturate the whole machine — fine on a dedicated show box,
bad on a laptop that still has to stay responsive (and on a small
machine it can make everything, including this control panel, crawl).

limit_threads() caps both to about a third of the cores. ONNX Runtime bakes
its thread count into a session at creation, so session_options() is
consulted when a model loads — the pipeline drops its cached models when
the setting changes so they reload with the new budget. That path covers
the expensive models (RVM, MODNet, SCRFD).

Caveat worth knowing: OpenCV's cap depends on the threading backend it
was built with. TBB/OpenMP/pthreads builds (typically Windows) honour an
arbitrary count; macOS wheels use GCD, which ignores everything except
0 (single-threaded). cv_threads() reports what actually took effect
rather than what we asked for.
"""
from __future__ import annotations

import os

_limited = False


def cores() -> int:
    return os.cpu_count() or 4


def budget() -> int:
    """Threads to allow when limited (0 = library default = all cores).

    Measured on a 12-core M2 Max with RVM (the heaviest model): letting
    ONNX Runtime loose used ~5.5 cores for 28 inferences/s, while 4
    threads gave 22/s for 3.6 cores and 3 threads gave 16/s for 2.5.
    The pipeline only needs ~15/s (30 fps, segmenting every other
    frame), so a third of the cores keeps full frame rate at well under
    half the CPU. Floored at 2 so small machines still get parallelism,
    capped at 6 because more buys almost nothing."""
    return max(2, min(6, round(cores() * 0.35))) if _limited else 0


def cv_threads() -> int:
    """OpenCV's actual thread count right now (0 if OpenCV is missing)."""
    try:
        import cv2
        return int(cv2.getNumThreads())
    except Exception:
        return 0


def limit_threads(enabled: bool) -> int:
    """Apply the cap to OpenCV now; returns the thread budget in effect."""
    global _limited
    _limited = bool(enabled)
    n = budget()
    try:
        import cv2
        cv2.setNumThreads(n if _limited else -1)  # -1 restores the default
    except Exception:
        pass
    return n


def session_options():
    """ONNX Runtime SessionOptions honouring the budget (None if unset)."""
    n = budget()
    if not n:
        return None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = max(1, n // 2)
        return so
    except ImportError:
        return None
