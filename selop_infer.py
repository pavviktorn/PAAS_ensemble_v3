#!/usr/bin/env python3.12
"""SeLop inference.

Single-image (default usage):
    python3.12 infer.py --ckpt runs/selop_default/best.pt --image /path/to/face.jpg

Full evaluation on a MIDS json:
    python3.12 infer.py --ckpt runs/selop_default/best.pt --val /path/to/mids.json

Labels for evaluation are derived via `get_label_all` (path-based). The decision
threshold defaults to the checkpoint's stored `threshold@real95` (the operating
point that maximises fake-recall while keeping real-recall >= 95%), or 0.5 if
absent; override with --threshold.
"""

import argparse
import json
import os

import torch

from selop.data import MidsBinaryDataset, build_index, build_transforms
from selop.engine import evaluate, fake_probability
from selop.metrics import format_metrics
from selop.model import SeLopModel
from selop.utils import amp_dtype_from_str, set_cpu_threads


def load_model(ckpt_path, clip_path, device, amp_dtype):
    ck = torch.load(ckpt_path, map_location="cpu")
    mc = ck["config"]
    model = SeLopModel(clip_path, num_classes=mc["num_classes"], rank=mc["rank"],
                       n_intervene=mc["n_intervene"]).to(device)
    model.load_trainable(ck)
    model.eval()
    return model, ck, mc


def resolve_threshold(args, ck):
    if args.threshold is not None:
        return args.threshold
    m = ck.get("metrics", {})
    for k in m:
        if k.startswith("threshold@real"):
            return float(m[k])
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--image", default=None, help="single image path")
    ap.add_argument("--val", default=None, help="MIDS json to evaluate (defaults to config val_data with --eval)")
    ap.add_argument("--eval", action="store_true", help="evaluate config's val_data")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    set_cpu_threads(cfg.get("omp_threads", 12))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = amp_dtype_from_str(cfg.get("amp_dtype", "bf16"))

    model, ck, mc = load_model(args.ckpt, cfg["clip_path"], device, amp_dtype)
    thr = resolve_threshold(args, ck)
    num_classes = mc["num_classes"]
    image_size = cfg.get("image_size", 336)

    # ---------- single-image inference ----------
    if args.image:
        from PIL import Image
        tfm = build_transforms(image_size, train=False)
        img = Image.open(args.image).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype,
                                              enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            logits = model(x)
        p_fake = float(fake_probability(logits, num_classes)[0])
        verdict = "FAKE" if p_fake >= thr else "REAL"
        print(json.dumps({
            "image": args.image,
            "p_fake": round(p_fake, 6),
            "threshold": round(thr, 6),
            "verdict": verdict,
        }, indent=2))
        return

    # ---------- full evaluation ----------
    val_json = args.val or (cfg["val_data"] if args.eval else None)
    if not val_json:
        ap.error("provide --image, or --val <json>, or --eval")
    samples = build_index(val_json, num_classes)
    tfm = build_transforms(image_size, train=False)
    ds = MidsBinaryDataset(samples, tfm, image_size, return_index=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    m = evaluate(model, loader, device, amp_dtype, num_classes,
                 is_dist=False, world_size=1,
                 real_recall_target=cfg.get("real_recall_target", 0.95))
    print(f"checkpoint: {args.ckpt}  (step {ck.get('step')}, epoch {ck.get('epoch')})")
    print(f"eval on: {val_json}")
    print(format_metrics(m))


if __name__ == "__main__":
    main()
