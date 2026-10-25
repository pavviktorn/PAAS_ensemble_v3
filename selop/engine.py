"""Shared evaluation routine (used by train.py and infer.py)."""

import numpy as np
import torch

from .data import class_names
from .get_label import REAL
from .metrics import compute_metrics
from .utils import all_gather_object


@torch.no_grad()
def fake_probability(logits, num_classes):
    """P(fake) from class logits. Binary: softmax[:,1]. 4-class: 1 - softmax[:,REAL]."""
    p = torch.softmax(logits.float(), dim=-1)
    if num_classes == 2:
        return p[:, 1]
    return 1.0 - p[:, REAL]


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes,
             is_dist=False, world_size=1, real_recall_target=0.95):
    """Run the model over `loader` (which must yield (x, label, idx)) and return
    a metrics dict. Distributed-safe: gathers per-sample (idx, score, label) and
    de-duplicates the DistributedSampler padding so the metric is exact."""
    model.eval()
    local = {}  # idx -> (fake_score, true_multiclass, pred_multiclass)
    for x, label, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits = model(x)
        score = fake_probability(logits, num_classes).cpu().numpy()
        pred = logits.float().argmax(dim=-1).cpu().numpy()
        label = label.numpy()
        idx = idx.numpy()
        for j in range(len(idx)):
            local[int(idx[j])] = (float(score[j]), int(label[j]), int(pred[j]))

    gathered = all_gather_object(local, is_dist, world_size)
    merged = {}
    for d in gathered:
        merged.update(d)
    ks = list(merged.keys())
    scores = np.array([merged[k][0] for k in ks], dtype=np.float64)
    true_mc = np.array([merged[k][1] for k in ks], dtype=np.int64)
    pred_mc = np.array([merged[k][2] for k in ks], dtype=np.int64)
    bin_labels = (true_mc != REAL).astype(np.int64)  # real=0 vs fake=1
    return compute_metrics(scores, bin_labels, real_recall_target=real_recall_target,
                           true_multiclass=true_mc, pred_multiclass=pred_mc,
                           class_names=class_names(num_classes))
