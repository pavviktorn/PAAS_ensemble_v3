"""Binary metrics (ACC / AUC / AP) + cross-set sACC.

FFAA's headline robustness metric is ``sACC`` -- the standard deviation of per-dataset accuracy
across the OW-FFA-Bench test sets (lower = better generalisation).  ``cross_set_summary`` computes
it.  Metrics use numpy directly (sklearn is used if available but not required) so they run in the
dependency-light smoke environment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) == 0:
        return float("nan")
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    return float((yt == yp).mean())


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(yt, ys))
    except Exception:
        # Mann-Whitney U statistic with tie handling.
        order = np.argsort(ys, kind="mergesort")
        ranks = np.empty(len(ys), dtype=float)
        sorted_scores = ys[order]
        i = 0
        while i < len(ys):
            j = i
            while j + 1 < len(ys) and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        n_pos = len(pos)
        n_neg = len(neg)
        sum_ranks_pos = ranks[yt == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    if yt.sum() == 0:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(yt, ys))
    except Exception:
        order = np.argsort(-ys, kind="mergesort")
        yt = yt[order]
        tp = np.cumsum(yt)
        fp = np.cumsum(1 - yt)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / yt.sum()
        ap = 0.0
        prev_recall = 0.0
        for p, r in zip(precision, recall):
            ap += p * (r - prev_recall)
            prev_recall = r
        return float(ap)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    return {
        "acc": accuracy(y_true, y_pred),
        "auc": roc_auc(y_true, y_score),
        "ap": average_precision(y_true, y_score),
    }


def cross_set_summary(per_set_acc: Dict[str, float]) -> Dict[str, float]:
    accs = [a for a in per_set_acc.values() if not math.isnan(a)]
    if not accs:
        return {"mean_acc": float("nan"), "sacc": float("nan")}
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    # report sACC on a 0-100 scale, as in the FFAA paper
    return {"mean_acc": mean, "sacc": math.sqrt(var) * 100.0}
