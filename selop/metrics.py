"""Metrics for binary real/fake detection.

`scores` = predicted probability of the FAKE class (label 1).
`labels` = ground truth (1 = fake, 0 = real).

Reports AUC, EER, accuracy, and — tied to the project goal of high fake recall
while protecting the real class — the best fake-recall achievable at a chosen
minimum real-recall (default 95%) and the threshold that achieves it.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve)


def compute_metrics(scores, labels, real_recall_target=0.95,
                    true_multiclass=None, pred_multiclass=None, class_names=None):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    out = {"n": int(len(labels)), "n_real": int((labels == 0).sum()),
           "n_fake": int((labels == 1).sum())}

    # headline metrics (binary real-vs-fake, fake = positive): acc / auc / ap
    pred = (scores >= 0.5).astype(int)
    out["acc"] = float((pred == labels).mean())
    both_classes = (labels == 0).any() and (labels == 1).any()
    if both_classes:
        out["bin_auc"] = float(roc_auc_score(labels, scores))  # binary real-vs-fake AUC
        out["ap"] = float(average_precision_score(labels, scores))
        fpr, tpr, thr = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[i] + fnr[i]) / 2.0)
        out["eer_threshold"] = float(thr[i])

    out["real_recall@0.5"] = float((scores[labels == 0] < 0.5).mean()) if (labels == 0).any() else 0.0
    out["fake_recall@0.5"] = float((scores[labels == 1] >= 0.5).mean()) if (labels == 1).any() else 0.0

    # per-class recall (multiclass), if argmax predictions were provided
    if true_multiclass is not None and pred_multiclass is not None:
        t = np.asarray(true_multiclass, dtype=np.int64)
        p = np.asarray(pred_multiclass, dtype=np.int64)
        names = class_names or [str(c) for c in sorted(set(t.tolist()))]
        for c, name in enumerate(names):
            m = t == c
            if m.any():
                out[f"recall_{name}"] = float((p[m] == c).mean())

    # Max fake-recall subject to real-recall >= target. A sample is predicted
    # real iff score < t, so real-recall(t) is non-decreasing and fake-recall(t)
    # is non-increasing in t -> pick the smallest t meeting the real-recall floor.
    real = scores[labels == 0]
    fake = scores[labels == 1]
    tag = int(round(real_recall_target * 100))
    if real.size and fake.size:
        cands = np.unique(np.concatenate([scores, [scores.max() + 1e-6]]))
        best_t, best_fr, best_rr = None, -1.0, None
        for t in cands:
            rr = float((real < t).mean())
            if rr >= real_recall_target:
                fr = float((fake >= t).mean())
                if fr > best_fr:
                    best_t, best_fr, best_rr = float(t), fr, rr
        if best_t is not None:
            out[f"fake_recall@real{tag}"] = best_fr
            out[f"threshold@real{tag}"] = best_t
            out[f"real_recall@real{tag}"] = best_rr
    return out


def format_metrics(m):
    keys = ["n", "n_real", "n_fake", "acc", "bin_auc", "ap", "eer", "eer_threshold",
            "real_recall@0.5", "fake_recall@0.5"]
    pc = [k for k in m if k.startswith("recall_")]
    extra = pc + [k for k in m if k.startswith(("fake_recall@real", "threshold@real", "real_recall@real"))]
    lines = []
    for k in keys + extra:
        if k in m:
            v = m[k]
            lines.append(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    return "\n".join(lines)
