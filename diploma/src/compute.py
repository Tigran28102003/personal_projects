"""Hardware detection + thread budgeting + utilisation logging (W1).

The GBM grid (9 studies = 3 setups × 3 target_types, × Optuna-trials × WF-folds) on a
small dataset (~70k×~50) is a **CPU-parallel throughput** problem, NOT a GPU one. The
budgeting rule is a single invariant: ``parallel_fits × lgbm_threads ≤ n_physical`` — many
single-threaded fits in parallel beat a few multi-threaded ones on small data. The P100
GPU is reserved for the neural nets (``nn_models``).

This module is import-light and degrades gracefully: ``psutil``/``torch``/``pynvml`` are
all optional. Detection is overridable by env (``COMPUTE_MODE``, ``PARALLEL_FITS``,
``LGBM_THREADS``, ``NN_DEVICE``) so the same code runs on this 8-core macOS box and the
36-core + P100 Linux box.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _cpu_counts() -> tuple[int, int]:
    """(n_physical, n_logical); psutil if present, else os.cpu_count fallback."""
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:  # noqa: BLE001 — psutil optional
        pass
    return int(physical), int(logical)


def _detect_gpu() -> Optional[str]:
    """CUDA device name if present, else None. **Deliberately torch-free** (pynvml /
    nvidia-smi): the GBM/Optuna worker processes import this, and importing torch
    alongside LightGBM risks a duplicate-libomp segfault on macOS. torch is imported only
    inside ``nn_models`` (its own subprocess)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    except Exception:  # noqa: BLE001 — pynvml optional
        pass
    try:
        import shutil
        import subprocess
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def detect() -> dict:
    """Compute profile. Small-data default: ``lgbm_threads=1, parallel_fits=n_physical``
    (clamped so ``parallel_fits × lgbm_threads ≤ n_physical``). Env overrides honoured."""
    n_physical, n_logical = _cpu_counts()
    gpu = _detect_gpu()
    mode = os.environ.get("COMPUTE_MODE", "DETERMINISTIC").upper()
    lgbm_threads = int(os.environ.get("LGBM_THREADS", "1"))
    default_pf = max(1, n_physical // max(1, lgbm_threads))
    parallel_fits = int(os.environ.get("PARALLEL_FITS", str(default_pf)))
    # enforce the budget invariant
    if parallel_fits * lgbm_threads > n_physical:
        parallel_fits = max(1, n_physical // max(1, lgbm_threads))
    return {
        "n_physical": n_physical, "n_logical": n_logical, "gpu": gpu,
        "parallel_fits": parallel_fits, "lgbm_threads": lgbm_threads,
        "mode": mode if mode in ("DETERMINISTIC", "FAST") else "DETERMINISTIC",
        "nn_device": os.environ.get("NN_DEVICE", ""),   # "" => auto (cuda>mps>cpu)
    }


def set_worker_threads(n: int) -> None:
    """Pin BLAS/OMP thread counts in the current (worker) process — anti-oversubscription
    when many fits run in parallel. Call at the top of each pool worker."""
    n = str(int(max(1, n)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n


def log_utilization(tag: str = "") -> dict:
    """Snapshot CPU/RAM (+GPU if pynvml) utilisation to the log. Returns the dict."""
    info = {}
    try:
        import psutil
        info["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        info["ram_pct"] = psutil.virtual_memory().percent
    except Exception:  # noqa: BLE001
        pass
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        info["gpu_mem_pct"] = 100 * pynvml.nvmlDeviceGetMemoryInfo(h).used / pynvml.nvmlDeviceGetMemoryInfo(h).total
    except Exception:  # noqa: BLE001 — pynvml optional / no GPU
        pass
    logger.info("utilisation%s: %s", f"[{tag}]" if tag else "", info)
    return info


def setup_logging(tag: str = "run", level: int = logging.INFO) -> logging.Logger:
    """Console INFO + ``logs/run_<tag>.log`` file handler (idempotent). Process-safe
    enough for the main process; pool workers log to console (loky captures)."""
    try:
        from . import config
    except ImportError:
        import config
    logdir = config.ROOT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
    have_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                       for h in root.handlers)
    if not have_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)
    fpath = logdir / f"run_{tag}.log"
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(fpath)
               for h in root.handlers):
        fh = logging.FileHandler(fpath)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root
