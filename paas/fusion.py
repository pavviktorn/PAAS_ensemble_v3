"""Component-level fusion of the per-frame fake-scores of the v3 detectors.

Each component emits a per-frame P(fake) in [0,1]:
  * ffaa               : FFAA make_decision forgery_score (match if pred==fake else 1-match)
  * A1_9c / A2_9c / A3_9c : each 9-class member's own fake-score (1 - P(real) marginal)
  * gsd                : GSD 3-class 1 - P(real)
  * selop              : SeLop 3-class 1 - P(real)

v3 fuses a chosen subset by PLAIN MEAN (recommended) or a WEIGHTED mean. Operating thresholds for
the recommended mean-of-5 {ffaa,A1_9c,A2_9c,gsd,selop} are measured on axonlabs_data_1 (Exp 13):

    OPERATING_POINTS  (real-recall floor -> fused-score threshold, fake-recall achieved)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

METHODS = ("mean", "weighted")

# mean-of-5 {ffaa,A1_9c,A2_9c,gsd,selop} on axonlabs_data_1 (613,415 frames): threshold @ real floor.
OPERATING_POINTS = {
    "real80": {"tau": 0.1167, "fake_recall": 99.99},
    "real85": {"tau": 0.1596, "fake_recall": 99.98},
    "real90": {"tau": 0.1982, "fake_recall": 99.97},   # v3 default
    "real95": {"tau": 0.2633, "fake_recall": 99.89},
    "real98": {"tau": 0.3730, "fake_recall": 99.77},
    "real99": {"tau": 0.4492, "fake_recall": 99.58},
}


def _as_arr(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def fuse_components(comp_scores: Dict[str, np.ndarray], cfg) -> np.ndarray:
    """Fuse per-frame component fake-score arrays into one fused array.

    ``comp_scores`` maps component-name -> per-frame array; it must contain every name in
    ``cfg.components``. ``cfg`` is a FusionCfg (method + components + optional weights).
    """
    comps: List[str] = list(cfg.components)
    missing = [c for c in comps if c not in comp_scores]
    if missing:
        raise ValueError(f"fusion is missing component scores for {missing} "
                         f"(have {sorted(comp_scores)})")
    cols = [_as_arr(comp_scores[c]) for c in comps]
    n = cols[0].shape[0]
    for c, col in zip(comps, cols):
        if col.shape[0] != n:
            raise ValueError(f"component '{c}' has {col.shape[0]} scores, expected {n}")
    M = np.vstack(cols)                              # (n_components, n_frames)

    if cfg.method == "mean":
        return M.mean(axis=0)
    if cfg.method == "weighted":
        w = np.asarray(cfg.weights, dtype=np.float64)
        w = w / w.sum()
        return (w[:, None] * M).sum(axis=0)
    raise ValueError(f"unknown fusion method {cfg.method!r}; expected one of {METHODS}")


def fuse_scalar_components(comp_scores: Dict[str, float], cfg) -> float:
    """Single-frame convenience wrapper around :func:`fuse_components`."""
    arrs = {k: [v] for k, v in comp_scores.items()}
    return float(fuse_components(arrs, cfg)[0])
