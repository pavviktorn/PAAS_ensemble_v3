#!/usr/bin/env python3.12
"""Offline combination frontier from two results files (no model re-run).

Pairs a 9-class results file and an FFAA results file on image path and prints, at each
real-recall floor, the fake-recall + threshold for each model alone and for mean/weighted/max/min
fusion. This is the cheap engine for exploring "all combinations".

Usage:
  python3.12 scripts/combine_eval.py --ensemble results_ensemble.txt --ffaa results_1.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paas.io_results import frontier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True, help="9-class results file (results_ensemble.txt)")
    ap.add_argument("--ffaa", required=True, help="FFAA results file in the same line format (results_1.txt)")
    ap.add_argument("--floors", default="0.80,0.85,0.90,0.95,0.98")
    ap.add_argument("--weights", default="0.2,0.5", help="ensemble weights to tabulate for weighted fusion")
    args = ap.parse_args()
    floors = tuple(float(x) for x in args.floors.split(","))
    weights = tuple(float(x) for x in args.weights.split(","))

    r = frontier(args.ensemble, args.ffaa, floors=floors, ensemble_weights=weights)
    print(f"paired frames: {r['n_paired']}  real={r['n_real']}  fake={r['n_fake']}\n")
    names = list(r["methods"].keys())
    hdr = f"{'real floor':>10s} | " + " | ".join(f"{n:>16s}" for n in names)
    print(hdr); print("-" * len(hdr))
    for fl in floors:
        cells = []
        for n in names:
            c = r["methods"][n][fl]
            cells.append(f"{c['fake_rec']*100:6.2f}% @t{c['tau']:.3f}")
        print(f"{fl*100:9.0f}% | " + " | ".join(f"{c:>16s}" for c in cells))
    print("\n(cell = fake-recall @ the threshold hitting that real-recall floor)")


if __name__ == "__main__":
    main()
