"""Evaluate a MIDS++ checkpoint, OW-FFA-Bench style.

Pass one or more test sets; the script reports per-set ACC/AUC/AP and the cross-set summary
(mean ACC and ``sACC`` -- the std of per-set ACC, FFAA's headline robustness metric).

    mids-pp-eval --checkpoint runs/mids_pp/best.pt \\
        --data DPF:/data/dpf.json DFD:/data/dfd.json DFDC:/data/dfdc.json
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import MidsAnswersDataset, build_image_transform, collate_fn
from .metrics import binary_metrics, cross_set_summary
from .selector import make_decision_batch
from .tokenize import build_tokenizer


def _parse_specs(specs: List[str]) -> List[Tuple[str, str]]:
    out = []
    for spec in specs:
        if ":" in spec and not os.path.exists(spec):
            name, path = spec.split(":", 1)
        else:
            name, path = os.path.splitext(os.path.basename(spec))[0], spec
        out.append((name, path))
    return out


@torch.no_grad()
def evaluate_one(model, cfg, tokenizer, transform, path: str, device, batch_size: int) -> Dict[str, float]:
    ds = MidsAnswersDataset(path, cfg, "val", transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)
    s = cfg.samples_per_image
    y_true, y_pred, y_score = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        enc = tokenizer(batch["texts"], return_tensors="pt", padding="longest", truncation=True, max_length=512).to(device)
        b = images.size(0)
        out = model(enc, images, None, b, s - 1, 0)
        scores = F.softmax(out["logits"].float(), dim=2)[:, :s, :]
        _, preds, _, forgery = make_decision_batch(batch["answers_result"], scores, chunk_size=s)
        y_true.extend(batch["cls_label"].tolist())
        y_pred.extend(preds)
        y_score.extend(forgery)
    return binary_metrics(y_true, y_pred, y_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIDS++")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="[name:]path JSON files")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides for loading")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        k, v = item.split("=", 1)
        overrides[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_checkpoint(args.checkpoint, device=device, overrides=overrides)
    tokenizer = build_tokenizer(cfg)
    transform = build_image_transform(cfg, "val")

    per_set_acc: Dict[str, float] = {}
    for name, path in _parse_specs(args.data):
        m = evaluate_one(model, cfg, tokenizer, transform, path, device, args.batch_size)
        per_set_acc[name] = m["acc"]
        print(f"{name:>10s}  ACC={m['acc']*100:5.2f}  AUC={m['auc']*100:5.2f}  AP={m['ap']*100:5.2f}")

    summary = cross_set_summary(per_set_acc)
    print("-" * 44)
    print(f"{'MEAN':>10s}  ACC={summary['mean_acc']*100:5.2f}  sACC={summary['sacc']:5.2f}")


if __name__ == "__main__":
    main()
