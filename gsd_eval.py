#!/usr/bin/env python3.12
"""Evaluate a trained GSD checkpoint on a real/fake tree or json list.

  python3.12 eval.py --ckpt runs/gsd/best.pt --data /path/val [--set batch_size=64]
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsd.config import GSDConfig
from gsd.model import GSDDetector
from gsd.data import FaceDataset
from gsd.engine import evaluate


def load_model(ckpt_path: str, device: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = GSDConfig.from_dict(payload["config"])
    model = GSDDetector(cfg)
    model.trainable.load_state_dict(payload["trainable"])
    model.head.load_state_dict(payload["head"])
    if payload.get("anchor_U") is not None:                  # fixed anchor embedded at train time
        model.set_fixed_U(payload["anchor_U"].to(device))
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser(description="Evaluate GSD detector")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    ds = FaceDataset(args.data, cfg.image_size, train=False, cfg=cfg, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)
    ev = evaluate(model, dl, device, cfg)
    from gsd.config import CLASS_NAMES
    names = CLASS_NAMES if cfg.num_classes == 3 else ("real", "fake")
    rec = "  ".join(f"{n}: recall={r:.4f} (n={c})" for n, r, c in zip(names, ev["recall"], ev["counts"]))
    print(f"real-vs-fake AUC={ev['bin_auc']:.4f} | acc={ev['acc']:.4f} | bal_acc={ev['bal_acc']:.4f} "
          f"| n={ev['n']}")
    print("  " + rec)


if __name__ == "__main__":
    main()
