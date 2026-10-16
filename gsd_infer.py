#!/usr/bin/env python3.12
"""Score images with a trained GSD detector.

GSD estimates the semantic subspace from the *batch*. Checkpoints trained by train.py have a fixed
semantic basis U **embedded** (built from the testset), so single-image scoring works out of the box.
Override the embedded anchor if needed:
  --anchor      a precomputed anchor.pt  (from build_anchor.py)
  --anchor-dir  a folder of reference images (U recomputed on the fly each call)

  python3.12 infer.py --ckpt runs/gsd/best.pt --input one.jpg          # uses embedded anchor
  python3.12 infer.py --ckpt runs/gsd/best.pt --input /path/imgs [--anchor runs/gsd/anchor.pt]
"""
import argparse
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsd.data import build_transform, IMG_EXT, resolve_label
from gsd.config import CLASS_NAMES
from eval import load_model


def gather(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for dp, _, files in os.walk(path):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append(os.path.join(dp, fn))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="GSD inference")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--anchor", default=None, help="precomputed anchor.pt (from build_anchor.py)")
    ap.add_argument("--anchor-dir", default=None, help="reference images to freeze the semantic basis")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None, help="write per-image JSONL here")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)
    tf = build_transform(cfg.image_size, train=False, cfg=cfg)

    if args.anchor:                                       # override: load a precomputed, deterministic U
        a = torch.load(args.anchor, map_location="cpu")
        if a["k"] != cfg.k or a["guide_pool"] != cfg.guide_pool:
            print(f"[gsd] WARNING: anchor (k={a['k']}, guide={a['guide_pool']}) != model "
                  f"(k={cfg.k}, guide={cfg.guide_pool})", flush=True)
        model.set_fixed_U(a["U"].to(device))
        print(f"[gsd] fixed anchor loaded from {args.anchor} "
              f"(U{tuple(a['U'].shape)}, built from {a.get('n_ref','?')} refs)", flush=True)
    elif args.anchor_dir:                                 # override: recompute U from reference images
        apaths = gather(args.anchor_dir)[: max(2, args.batch_size)]
        ax = torch.stack([tf(Image.open(p).convert("RGB")) for p in apaths]).to(device)
        from gsd.model import _pool
        guides = _pool(model.frozen(ax).last_hidden_state, cfg.guide_pool)
        model.set_fixed_anchor(guides)
        print(f"[gsd] fixed anchor from {len(apaths)} reference images", flush=True)
    elif model._fixed_U is not None:                      # default: anchor embedded in the checkpoint
        print(f"[gsd] using embedded anchor U{tuple(model._fixed_U.shape)} (from training)", flush=True)
    else:
        print("[gsd] WARNING: no anchor; single images (batch=1) run with GSD disabled "
              "(use --anchor / --anchor-dir, or score >=2 images at once)", flush=True)

    names = list(CLASS_NAMES) if cfg.num_classes == 3 else ["real", "fake"]
    paths = gather(args.input)
    fh = open(args.out, "w") if args.out else None
    labels, scores = [], []                                 # binary real(0)/fake(1) for AUC
    for i in range(0, len(paths), args.batch_size):
        batch = paths[i:i + args.batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        probs = torch.softmax(model(x).float(), dim=1).cpu().tolist()
        for p, pr in zip(batch, probs):
            fake = 1.0 - pr[0]                              # class 0 == real
            cls = int(max(range(len(pr)), key=lambda j: pr[j]))
            rec = {"image": p, "pred": names[cls], "fake_prob": round(fake, 6),
                   "probs": {n: round(v, 6) for n, v in zip(names, pr)}}
            print(json.dumps(rec))
            if fh:
                fh.write(json.dumps(rec) + "\n")
            lab = resolve_label(p, cfg.num_classes)
            if lab is not None:
                labels.append(0 if lab == 0 else 1); scores.append(fake)
    if fh:
        fh.close()
    if labels:
        from gsd.engine import roc_auc
        acc = sum((s >= 0.5) == bool(y) for s, y in zip(scores, labels)) / len(labels)
        print(f"\n[gsd] labelled subset: real-vs-fake AUC={roc_auc(labels, scores):.4f} "
              f"acc={acc:.4f} n={len(labels)}")


if __name__ == "__main__":
    main()
