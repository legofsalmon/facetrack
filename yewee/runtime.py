"""CPU budget for the ML runtimes.

OpenCV and ONNX Runtime both default to "use every core", so a busy
pipeline can saturate the whole machine — fine on a dedicated show box,
bad on a laptop that still has to stay responsive (and on a small
machine it can make everything, including this control panel, crawl).

limit_threads() caps both to about a third of the cores. ONNX Runtime bakes
its thread count into a session at creation, so session_options() is
consulted when a model loads — the pipeline drops its cached models when
the setting changes so they reload with the new budget. That path covers
the expensive models (MODNet, CenterFace, and RVM in internal builds).

Caveat worth knowing: OpenCV's cap depends on the threading backend it
was built with. TBB/OpenMP/pthreads builds (typically Windows) honour an
arbitrary count; macOS wheels use GCD, which ignores everything except
0 (single-threaded). cv_threads() reports what actually took effect
rather than what we asked for.
"""
from __future__ import annotations

import logging
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
    frame), so a third of the cores holds the frame rate at well under
    half the CPU: 2 cores -> 1, 4 -> 2, 8 -> 3, 12 -> 4. Capped at 6
    because more buys almost nothing, and always at least one below the
    core count so even a small machine keeps something in reserve —
    that's the case this setting exists for."""
    return max(1, min(6, (cores() + 2) // 3)) if _limited else 0


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


#: Tried in order, per platform. TensorRT is deliberately not here: it
#: needs a matching TensorRT install on top of CUDA, and it compiles an
#: engine per model and input shape on first use — a stall of a minute or
#: more, which is the last thing you want when a show is about to start.
#: Pass it explicitly if you want it.
#:
#: macOS: CoreML first, which puts the silhouette and matting models on
#: the Apple GPU / Neural Engine of an M-series machine. Windows: CUDA
#: (onnxruntime-gpu, NVIDIA) first, then DirectML (onnxruntime-directml,
#: any AMD / Intel / NVIDIA card through DirectX 12). Only one of those
#: two packages can be installed at a time — the Windows launcher picks
#: by looking for an NVIDIA card — so in practice a machine lists one.
#: The CPU provider is always last and always present.
_PROVIDERS_BY_PLATFORM = {
    "darwin": ("CoreMLExecutionProvider", "CPUExecutionProvider"),
    "win32": ("CUDAExecutionProvider", "DmlExecutionProvider",
              "CPUExecutionProvider"),
}

#: Options worth setting for a provider when it is asked for. CoreML's
#: ML Program format is the one Apple still develops (the older
#: NeuralNetwork format is frozen) and the one that reaches the Neural
#: Engine on current macOS.
_PROVIDER_OPTIONS = {
    "CoreMLExecutionProvider": {"ModelFormat": "MLProgram"},
}

#: Short names for the panel and the doctor.
PROVIDER_LABELS = {
    "CoreMLExecutionProvider": "Apple GPU / Neural Engine (CoreML)",
    "CUDAExecutionProvider": "NVIDIA GPU (CUDA)",
    "TensorrtExecutionProvider": "NVIDIA GPU (TensorRT)",
    "DmlExecutionProvider": "GPU (DirectML)",
    "CPUExecutionProvider": "CPU",
}


def platform_providers(platform: str | None = None) -> tuple[str, ...]:
    """The provider chain for a platform (sys.platform by default)."""
    import sys
    return _PROVIDERS_BY_PLATFORM.get(
        platform or sys.platform, ("CUDAExecutionProvider", "CPUExecutionProvider"))


DEFAULT_PROVIDERS = platform_providers()

_accel = "auto"


def set_accelerator(mode: str) -> None:
    """'auto' = the platform chain; 'cpu' = the CPU provider only. The
    pipeline drops its cached models when this changes so they reload."""
    global _accel
    _accel = "cpu" if mode == "cpu" else "auto"


def accelerator() -> str:
    return _accel


def provider_label(name: str) -> str:
    return PROVIDER_LABELS.get(name, name.replace("ExecutionProvider", ""))


def make_session(model_path, providers=None, threads: int | None = None):
    """Open an ONNX Runtime session on the best provider that actually works.

    onnxruntime.get_available_providers() reports what the package was
    *compiled* with, not what can load on this machine. A Windows
    onnxruntime-gpu build happily lists TensorrtExecutionProvider and
    CUDAExecutionProvider on a machine with no CUDA at all, and asking for
    one then dies with a missing cublas64_12.dll. The only honest test is
    to try, so try in order and fall back to whatever loads.
    """
    import onnxruntime as ort

    listed = ort.get_available_providers()
    wanted = providers or (("CPUExecutionProvider",) if _accel == "cpu"
                           else DEFAULT_PROVIDERS)
    chain = [p for p in wanted if p in listed]
    if "CPUExecutionProvider" not in chain:
        chain.append("CPUExecutionProvider")   # always leave a way to run

    def _with_options(names):
        return [(n, _PROVIDER_OPTIONS[n]) if n in _PROVIDER_OPTIONS else n
                for n in names]

    log = logging.getLogger("yewee")
    last: Exception | None = None
    for i, provider in enumerate(chain):
        try:
            return ort.InferenceSession(str(model_path),
                                        sess_options=session_options(threads),
                                        providers=_with_options(chain[i:]))
        except Exception as exc:                        # noqa: BLE001
            last = exc
            nxt = chain[i + 1] if i + 1 < len(chain) else None
            log.warning("%s could not load (%s)%s", provider,
                        str(exc).strip().splitlines()[0][:160],
                        f"; falling back to {nxt}" if nxt else "")
    raise RuntimeError(f"no usable ONNX Runtime provider for {model_path}") from last


def active_provider(session) -> str:
    """The provider a session actually runs on (first in its list)."""
    try:
        return session.get_providers()[0]
    except Exception:                                   # noqa: BLE001
        return "CPUExecutionProvider"


_probed: str | None = None


def probe_provider(model_path) -> str:
    """Which provider this machine really runs models on, found by opening
    one small session (cached). This is what the 'auto' silhouette model
    choice and the panel's accelerator line are based on."""
    global _probed
    if _probed is None:
        try:
            _probed = active_provider(make_session(model_path))
        except Exception:                               # noqa: BLE001
            _probed = "CPUExecutionProvider"
    return _probed


def reset_probe() -> None:
    global _probed
    _probed = None


def session_options(threads: int | None = None):
    """ONNX Runtime SessionOptions: the thread budget (or an explicit
    count for a small model), and no spin-waiting.

    Spinning is ONNX Runtime's default: between ops its worker threads
    busy-wait rather than sleep, which is fastest for one model alone
    and ruinous beside a show loop — on a 4-core box the Selfie engine
    on its own thread took the loop from 54 fps to 23 while segmenting
    8 tiles, because its idle workers were burning the cores the loop
    needed. Off, the same run holds the frame rate."""
    n = threads if threads is not None else budget()
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if n:
            so.intra_op_num_threads = n
            so.inter_op_num_threads = max(1, n // 2)
        return so
    except ImportError:
        return None
