#!/usr/bin/env python3.12
"""PAAS_ensemble_v3 inference CLI.

Run the 5-detector face real/fake ensemble (FFAA + A1_9c + A2_9c + GSD + SeLop) on one or more
images. The detector set and fusion come from an experiment config; convenience flags override the
fusion components/method, threshold, and device without editing the config.

Examples:
  python3.12 inference.py face.jpg                                  # default config/experiments/paas5_mean.json
  python3.12 inference.py a.jpg b.png --json
  python3.12 inference.py --dir /path/to/folder
  python3.12 inference.py face.jpg --components gsd,selop           # a fast 2-detector subset
  python3.12 inference.py face.jpg --threshold 0.373               # real-98 operating point
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paas.config import PaasConfig, ALL_COMPONENTS
from paas.pipeline import PaasPipeline

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_CONFIG = os.path.join(ROOT, "config", "experiments", "paas5_mean.json")


def collect_images(images, directory):
    if images is None:
        images = []
    elif isinstance(images, str):
        images = [images] if images else []
    files = [p for p in images if p]
    if directory and os.path.isdir(directory):
        for dp, _, fns in os.walk(directory):
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    files.append(os.path.join(dp, fn))
    seen, out = set(), []
    for f in files:
        k = os.path.abspath(f)
        if k not in seen:
            seen.add(k); out.append(f)
    return out


def build_config(args) -> PaasConfig:
    cfg = PaasConfig.from_file(args.config)
    if args.device:
        cfg.device = args.device
    if args.components:
        comps = [c.strip() for c in args.components.split(",") if c.strip()]
        cfg.fusion.components = comps
    if args.method:
        cfg.fusion.method = args.method
    if args.weights:
        cfg.fusion.weights = [float(x) for x in args.weights.split(",")]
    if args.threshold is not None:
        cfg.decision.threshold = args.threshold
    if args.ambiguous_match_min is not None:
        cfg.decision.real_ambiguous_match_min = args.ambiguous_match_min
    # only load the detectors the requested components need
    need = cfg.needs()
    cfg.ffaa.enabled = need["ffaa"]
    cfg.ensemble9.enabled = need["ens"]
    cfg.gsd.enabled = need["gsd"]
    cfg.selop.enabled = need["selop"]
    return cfg.validate()


def main():
    ap = argparse.ArgumentParser(description="PAAS_ensemble_v3 5-detector inference")
    ap.add_argument("--images", default=os.path.join(ROOT, "images", "0034.jpg"), help="image file path(s)")
    ap.add_argument("--dir", default="", help="recurse a folder and score every image in it")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"experiment config (default: {DEFAULT_CONFIG})")
    ap.add_argument("--components", default="",
                    help=f"comma list overriding fusion.components; any of {','.join(ALL_COMPONENTS)}")
    ap.add_argument("--method", choices=("mean", "weighted"), default="", help="override fusion method")
    ap.add_argument("--weights", default="", help="comma weights for method=weighted (len==components)")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--ambiguous-match-min", type=float, default=None,
                    help="decision==real & match<this -> 'ambiguous' (default 0.9)")
    ap.add_argument("--device", default="", help="cuda:0 / cuda:2 / cpu (default from config)")
    ap.add_argument("--ens-batch", type=int, default=32)
    ap.add_argument("--ffaa-batch", type=int, default=8)
    ap.add_argument("--gsd-batch", type=int, default=64)
    ap.add_argument("--selop-batch", type=int, default=64)
    ap.add_argument("--json", action="store_true", help="emit a JSON array instead of human-readable lines")
    args = ap.parse_args()

    files = collect_images(args.images, args.dir)
    if not files:
        ap.error("no images given (pass image paths and/or --dir)")

    cfg = build_config(args)
    print(f"# loading config='{cfg.name}' fusion={cfg.fusion.method} components={cfg.fusion.components} "
          f"on {cfg.device} ... [{len(files)} image(s); FFAA 7B MLLM load can take a minute]", flush=True)
    t0 = time.time()
    pipe = PaasPipeline(cfg)
    load_t = time.time() - t0
    print(f"# models loaded in {load_t:.1f}s; scoring ...", flush=True)

    t1 = time.time()
    results = pipe.predict_images(files, ens_batch_size=args.ens_batch, ffaa_batch_size=args.ffaa_batch,
                                  gsd_batch_size=args.gsd_batch, selop_batch_size=args.selop_batch)
    dt = (time.time() - t1) / max(len(files), 1)
    for r in results:
        r["processing_time_sec"] = round(dt, 4)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"# PAAS_v3 config='{cfg.name}' fusion={cfg.fusion.method} threshold={cfg.decision.threshold} "
          f"components={cfg.fusion.components} | load={load_t:.1f}s, {dt*1000:.0f}ms/img")
    for r in results:
        img = r.get("image", "?")
        if r.get("decision") == "error":
            print(f"  ERROR  {img}  ({r.get('error')})")
            continue
        comp = r.get("components") or {}
        cstr = " ".join(f"{k}={v:.3f}" for k, v in comp.items())
        print(f"  {r['decision']:9s} {r['forgery_type']:8s} score={r['forgery_score']:.4f} "
              f"match={r['match_score']:.4f}  [{cstr}]  {img}")


if __name__ == "__main__":
    main()
