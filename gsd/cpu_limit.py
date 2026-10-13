"""Cap CPU thread usage (OpenMP / MKL / BLAS + torch intra-op) to a fraction of the cores.

Call `limit_cpu()` at the very top of an entrypoint -- BEFORE `import torch` -- so the OMP/MKL/BLAS
env vars are in place when those runtimes initialise (DataLoader workers inherit them too, preventing
oversubscription). It is safe to call again after torch is imported (e.g. with the value from the
config) to apply `torch.set_num_threads` exactly.

Default fraction is 0.5 (50%); override per-run with the GSD_CPU_FRACTION env var or cfg.cpu_fraction.
"""
from __future__ import annotations

import os

_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def n_cpus() -> int:
    """Cores actually available to this process (respects cgroup/cpuset affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def threads_for(fraction: float) -> int:
    return max(1, int(round(n_cpus() * float(fraction))))


def limit_cpu(fraction=None, verbose: bool = True) -> int:
    """Set OMP/MKL/BLAS env vars + torch thread count to `fraction` of the cores. Returns #threads."""
    if fraction is None:
        fraction = float(os.environ.get("GSD_CPU_FRACTION", "0.5"))
    n = threads_for(fraction)
    for v in _ENV_VARS:
        os.environ[v] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    if verbose:
        print(f"[gsd] CPU limit: {n}/{n_cpus()} threads (~{float(fraction)*100:.0f}%) | "
              f"OMP_NUM_THREADS={n}", flush=True)
    return n
