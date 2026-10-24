#!/usr/bin/env python3.12
"""End-to-end dataset evaluation for PAAS_ensemble_v2.

Walks a folder tree, scores every still image and every sampled video frame with the model(s) a
config selects, and writes results in the unified line format. When BOTH models are enabled it
writes three files in one pass:
  * results_ensemble.txt  -- 9-class MLLM-free per-frame fake-score
  * results_ffaa.txt      -- FFAA MLLM+MIDS per-frame fake-score
  * results_paas.txt      -- the fused decision for THIS config
Feed the first two to scripts/combine_eval.py to explore every other combination offline -- no
need to re-run the 7B model per experiment.

Ground truth = the `real`/`fake` path component. Optional `--filter-real` skips low-quality real
frames (logged `SK`, excluded from accuracy), mirroring build_real_filtered.py.

Usage:
  python3.12 scripts/run_dataset.py --config config/experiments/mean.json \
      --input-dir /path/to/data --out-dir runs/exp_mean --frame-stride 10 --limit 0
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paas.config import PaasConfig
from paas.pipeline import PaasPipeline
from paas.io_results import fmt_line
from paas.decision import eval_label

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def truth_of(path: str):
    parts = path.lower().split(os.sep)
    if "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    return None


def iter_media(root):
    for dp, _, files in os.walk(root):
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                yield os.path.join(dp, fn), "image"
            elif ext in VID_EXT:
                yield os.path.join(dp, fn), "video"


def frames_of(path, kind, stride):
    """Yield (frame_key, rgb_uint8) for an image or sampled video frames."""
    if kind == "image":
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            yield path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"{path}#frame={i:06d}", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="runs/exp")
    ap.add_argument("--frame-stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="max media files (0=all)")
    ap.add_argument("--ens-batch", type=int, default=32)
    ap.add_argument("--ffaa-batch", type=int, default=8)
    ap.add_argument("--filter-real", type=int, default=0,
                    help="1 = skip low-quality real frames via paas.data.face_filter (optional dep)")
    args = ap.parse_args()

    cfg = PaasConfig.from_file(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = PaasPipeline(cfg)
    has_ens, has_ffaa = pipe.ens is not None, pipe.ffaa is not None

    rfilter = None
    if args.filter_real:
        try:
            from paas.data.face_filter import FaceQualityFilter
            rfilter = FaceQualityFilter()
        except Exception as e:
            print(f"[run_dataset] --filter-real requested but face filter unavailable ({e}); "
                  f"reals will NOT be filtered.")

    f_ens = open(os.path.join(args.out_dir, "results_ensemble.txt"), "w") if has_ens else None
    f_ffaa = open(os.path.join(args.out_dir, "results_ffaa.txt"), "w") if has_ffaa else None
    f_paas = open(os.path.join(args.out_dir, "results_paas.txt"), "w")
    for fh, hdr in ((f_ens, "9-class ensemble"), (f_ffaa, "FFAA MLLM+MIDS"), (f_paas, f"PAAS fused [{cfg.fusion.method}]")):
        if fh:
            fh.write(f"# PAAS_ensemble_v2 {hdr} | config={cfg.name} | input={args.input_dir}\n")
            fh.write("# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image\n")

    tally = {}  # (truth, kind-agnostic) -> [correct, total]
    n_media = 0
    for path, kind in iter_media(args.input_dir):
        truth = truth_of(path)
        if truth is None:
            continue
        n_media += 1
        if args.limit and n_media > args.limit:
            break
        keys, rgbs = [], []
        for k, rgb in frames_of(path, kind, args.frame_stride):
            if truth == "real" and rfilter is not None and not rfilter.passes(rgb):
                if f_ens: f_ens.write(fmt_line("SK", truth, "skip", "lowqual", None, None, k) + "\n")
                if f_ffaa: f_ffaa.write(fmt_line("SK", truth, "skip", "lowqual", None, None, k) + "\n")
                f_paas.write(fmt_line("SK", truth, "skip", "lowqual", None, None, k) + "\n")
                continue
            keys.append(k); rgbs.append(rgb)
        if not rgbs:
            continue
        res = pipe.predict_frames(rgbs, ens_batch_size=args.ens_batch, ffaa_batch_size=args.ffaa_batch)
        for k, r in zip(keys, res):
            if r["decision"] == "error":
                for fh in (f_ens, f_ffaa, f_paas):
                    if fh: fh.write(fmt_line("ER", truth, "----", "error", None, None, k) + "\n")
                continue
            if f_ens is not None and r.get("ensemble_fake") is not None:
                ef = r["ensemble_fake"]
                f_ens.write(fmt_line("OK", truth, "fake" if ef >= 0.5 else "real", "-", ef, ef if ef >= .5 else 1 - ef, k) + "\n")
            if f_ffaa is not None and r.get("ffaa_fake") is not None:
                ff = r["ffaa_fake"]
                f_ffaa.write(fmt_line("OK", truth, r.get("ffaa_analysis", "-"), "-", ff, r.get("ffaa_match"), k) + "\n")
            # fused (this config) -> results_paas + accuracy (ambiguous counts as fake)
            pe = eval_label(r["decision"])
            ok = (pe == truth)
            tag = "OK" if ok else "XX"
            f_paas.write(fmt_line(tag, truth, r["decision"], r["forgery_type"],
                                  r["forgery_score"], r["match_score"], k) + "\n")
            t = tally.setdefault(truth, [0, 0]); t[0] += ok; t[1] += 1

    for fh in (f_ens, f_ffaa, f_paas):
        if fh: fh.close()

    print(f"\n=== PAAS fused accuracy [{cfg.fusion.method}] (ambiguous counts as fake) ===")
    for truth, (c, n) in sorted(tally.items()):
        print(f"  {truth:5s}: {c}/{n} = {100*c/max(n,1):.2f}%")
    print(f"results in {args.out_dir}/  (results_paas.txt + per-model files for offline combine_eval)")


if __name__ == "__main__":
    main()
