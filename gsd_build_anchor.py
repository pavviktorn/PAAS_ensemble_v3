#!/usr/bin/env python3.12
"""Precompute and save a fixed GSD semantic basis `U` from a reference image set.

GSD normally estimates U from each batch's frozen guides. For deterministic single-image inference,
freeze U *once* from a representative reference set and reuse it. U is built from the FROZEN CLIP
only -- it does NOT depend on the trained head/backbone -- so it depends solely on:
  reference images + k + guide_pool + qr_method + clip_path.

A larger, balanced reference set (a few hundred images spanning real/pad/deepfake) gives a more
stable subspace than any single 128-image training batch.

  python3.12 build_anchor.py --ckpt runs/gsd/best.pt --ref-dir /path/imgs --out runs/gsd/anchor.pt [--limit 512]
  python3.12 build_anchor.py --config configs/default.json --ref-dir /path/imgs --out anchor.pt
"""
import argparse
import os
import sys

import torch
from PIL import Image
from transformers import CLIPVisionModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsd.config import GSDConfig
from gsd.data import build_transform, IMG_EXT
from gsd.model import _pool
from gsd.householder import semantic_basis


def gather(path, limit=0):
    out = []
    if os.path.isfile(path):
        out = [path]
    else:
        for dp, _, files in os.walk(path):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    out.append(os.path.join(dp, fn))
    if limit and limit < len(out):                       # strided -> spread across the ref tree/classes
        stride = len(out) / limit
        out = [out[int(i * stride)] for i in range(limit)]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Precompute a fixed GSD semantic basis U")
    ap.add_argument("--ckpt", default=None, help="checkpoint to read the config from (k/guide_pool/qr/clip_path)")
    ap.add_argument("--config", default=None, help="config json (alternative to --ckpt)")
    ap.add_argument("--ref-dir", required=True, help="folder of reference images")
    ap.add_argument("--out", required=True, help="output anchor.pt path")
    ap.add_argument("--limit", type=int, default=512, help="#reference images to use (strided; 0=all)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if not (args.ckpt or args.config):
        raise SystemExit("provide --ckpt or --config")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.ckpt:
        cfg = GSDConfig.from_dict(torch.load(args.ckpt, map_location="cpu")["config"])
    else:
        cfg = GSDConfig.from_file(args.config)

    paths = gather(args.ref_dir, args.limit)
    if len(paths) < cfg.k:
        raise SystemExit(f"need >= k={cfg.k} reference images, got {len(paths)}")
    print(f"[anchor] {len(paths)} reference images | k={cfg.k} guide={cfg.guide_pool} qr={cfg.qr_method}", flush=True)

    frozen = CLIPVisionModel.from_pretrained(cfg.clip_path).eval().to(device)
    tf = build_transform(cfg.image_size, train=False, cfg=cfg)
    guides = []
    for i in range(0, len(paths), args.batch_size):
        batch = paths[i:i + args.batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        guides.append(_pool(frozen(x).last_hidden_state, cfg.guide_pool).float().cpu())
    G = torch.cat(guides, dim=0)                          # (N, D)
    U = semantic_basis(G, cfg.k, cfg.qr_method)           # (D, K)

    ortho = (U.t() @ U - torch.eye(cfg.k)).abs().max().item()
    payload = {"U": U, "k": cfg.k, "guide_pool": cfg.guide_pool, "qr_method": cfg.qr_method,
               "clip_path": cfg.clip_path, "image_size": cfg.image_size, "n_ref": len(paths)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[anchor] saved U {tuple(U.shape)} -> {args.out}  (orthonormality max|UᵀU-I|={ortho:.2e})", flush=True)


if __name__ == "__main__":
    main()
