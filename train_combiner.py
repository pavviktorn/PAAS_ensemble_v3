#!/usr/bin/env python3.12
"""Fit a learned score-combiner from two results files (no model re-run, dependency-free).

Trains a logistic regression  P(fake) = sigmoid(w_e*ensemble_fake + w_f*ffaa_fake + b)  on the
per-frame paired scores, and writes a JSON usable by fusion.method=='learned':

    {"type": "logistic", "w": [w_e, w_f], "b": b}

Usage:
  python3.12 train/train_combiner.py --ensemble results_ensemble.txt --ffaa results_ffaa.txt \
      --out config/learned_combiner.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paas.io_results import parse_results_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--ffaa", required=True)
    ap.add_argument("--out", default="config/learned_combiner.json")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--class-balance", type=int, default=1,
                    help="1 = weight real/fake equally (recommended; fakes dominate counts)")
    args = ap.parse_args()

    E, Fa = parse_results_file(args.ensemble), parse_results_file(args.ffaa)
    keys = sorted(E.keys() & Fa.keys())
    if not keys:
        raise SystemExit("no paired frames between the two files")
    y = np.array([E[k][0] for k in keys], np.float64)            # 1 = fake
    X = np.stack([np.array([E[k][1] for k in keys]),             # ensemble_fake
                  np.array([Fa[k][1] for k in keys])], axis=1)   # ffaa_fake
    n = len(y)

    # per-sample weights for class balance
    w = np.ones(n)
    if args.class_balance:
        pf = y.mean()
        w[y == 1] = 0.5 / max(pf, 1e-6)
        w[y == 0] = 0.5 / max(1 - pf, 1e-6)

    theta = np.zeros(3)  # [w_e, w_f, b]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    for _ in range(args.steps):
        z = Xb @ theta
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (w * (p - y)) / n
        theta -= args.lr * grad

    spec = {"type": "logistic", "w": [float(theta[0]), float(theta[1])], "b": float(theta[2])}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(spec, open(args.out, "w"), indent=2)

    # report training-set frontier at the fitted operating point
    p = 1.0 / (1.0 + np.exp(-(Xb @ theta)))
    real = y == 0
    tau = float(np.quantile(p[real], 0.90))
    print(f"fit on {n} frames (real={int(real.sum())}, fake={int((~real).sum())})")
    print(f"  learned: P(fake)=sigmoid({theta[0]:.3f}*ens + {theta[1]:.3f}*ffaa + {theta[2]:.3f})")
    print(f"  @ real-recall 90%: tau={tau:.4f}  fake-recall={(p[~real] >= tau).mean()*100:.2f}%")
    print(f"  wrote {args.out}  (use with fusion.method='learned', fusion.learned_path='{args.out}')")


if __name__ == "__main__":
    main()
