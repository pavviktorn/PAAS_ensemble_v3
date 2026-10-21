"""Unified results I/O + offline combination analysis.

Line format (identical to results_ensemble.txt / the converted FFAA results_1.txt), so existing
parsers and the prior analysis carry over verbatim:

    OK/XX/SK/ER  truth=..  pred=..  type=..  fake=..  match=..  <path>

`SK` = low-quality real excluded from accuracy; `ER` = model error. A frame's ground-truth label is
the `real`/`fake` component of its path. This module also parses two such files (e.g. the 9-class
and FFAA results), joins them per image, and sweeps fusion methods/thresholds -- the cheap way to
explore "all combinations" without re-running any model.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np

LINE = re.compile(
    r"^(OK|XX)\s+truth=(\w+)\s+pred=\S+\s+type=\S+\s+fake=([0-9.]+)\s+match=\S+\s+(.*)$")


def fmt_line(tag: str, truth: str, pred: str, ftype: str, fake, match, path: str) -> str:
    fs = "------" if fake is None else f"{float(fake):.4f}"
    ms = "------" if match is None else f"{float(match):.4f}"
    tf = f"type={ftype}"
    tf = tf + " " * max(2, 15 - len(tf))
    return f"{tag}  truth={truth}  pred={pred}  {tf}fake={fs} match={ms}  {path}"


def parse_results_file(path: str) -> Dict[str, Tuple[int, float]]:
    """Return {image_path: (is_fake(0/1), fake_score)} for OK/XX rows only."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            m = LINE.match(ln.rstrip("\n"))
            if m:
                _, truth, fake, img = m.groups()
                d[img] = (1 if truth == "fake" else 0, float(fake))
    return d


# ----------------------------------------------------------------------------------------------
# Offline combination frontier (paired on image path)
# ----------------------------------------------------------------------------------------------
def frontier(ensemble_file: str, ffaa_file: str, floors=(0.80, 0.85, 0.90, 0.95, 0.98),
             ensemble_weights=(0.2, 0.5)) -> dict:
    """Pair two results files on image path; report, at each real-recall floor, the fake-recall
    achieved by each model alone and by mean / weighted / max / min fusion (with the threshold)."""
    E, Fa = parse_results_file(ensemble_file), parse_results_file(ffaa_file)
    keys = E.keys() & Fa.keys()
    tr = np.array([E[k][0] for k in keys], bool)
    e = np.array([E[k][1] for k in keys], np.float64)
    f = np.array([Fa[k][1] for k in keys], np.float64)
    real = ~tr

    methods = {"ffaa": f, "ensemble": e, "mean": (e + f) / 2,
               "max": np.maximum(e, f), "min": np.minimum(e, f)}
    for w in ensemble_weights:
        methods[f"weighted_{w}"] = w * e + (1 - w) * f

    def fr_at(score, fl):
        tau = float(np.quantile(score[real], fl))
        return tau, float((score[real] < tau).mean()), float((score[tr] >= tau).mean())

    table = {}
    for name, sc in methods.items():
        table[name] = {fl: dict(zip(("tau", "real_rec", "fake_rec"), fr_at(sc, fl))) for fl in floors}
    return {"n_paired": int(len(keys)), "n_real": int(real.sum()), "n_fake": int(tr.sum()),
            "floors": list(floors), "methods": table}
