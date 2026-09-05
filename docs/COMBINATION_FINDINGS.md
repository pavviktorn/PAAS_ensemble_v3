# Combination findings (frame-level)

Paired on **614,044** evaluated frames of `axonlabs_data_1` (48,526 real / 565,518 fake), using
each model's per-frame fake-score (`pred`/`type` ignored). FFAA fake-score = `match` if
`analysis==fake` else `1-match` (≡ make_decision's `forgery_score`). Reproduce with:
`scripts/combine_eval.py --ensemble results_ensemble.txt --ffaa results_1.txt`.

## Score distributions
| model    | real mean/median | fake mean/median |
|----------|------------------|------------------|
| ensemble | 0.252 / 0.192    | 0.945 / 0.999    |
| FFAA     | 0.027 / 0.000    | 0.991 / 1.000    |

FFAA is sharply bimodal (reals≈0, fakes≈1) → far stronger single signal.

## Frame-level frontier — fake-recall @ threshold hitting each real-recall floor
|real floor|   FFAA   | ensemble      |   mean        | max(OR)       | min(AND)      | weighted 0.2e+0.8f |
|---: |---------------|---------------|---------------|---------------|---------------|---------------|
| 80% | 99.85% @t.000 | 97.89% @t.520 | 99.82% @t.287 | 99.79% @t.567 | 99.89% @t.000 | 99.83% @t.120 |
| 85% | 99.80% @t.001 | 96.76% @t.645 | 99.76% @t.331 | 99.71% @t.660 | 99.81% @t.000 | 99.79% @t.133 |
| 88% | 99.73% @t.001 | 95.99% @t.664 | 99.74% @t.333 | 99.64% @t.666 | 99.75% @t.001 | 99.78% @t.133 |
| 90% | 99.68% @t.002 | 95.47% @t.666 | 99.72% @t.334 | 99.57% @t.667 | 99.69% @t.001 | 99.73% @t.134 |
| 92% | 99.62% @t.005 | 94.86% @t.667 | 99.64% @t.343 | 99.47% @t.673 | 99.62% @t.002 | 99.66% @t.144 |
| 95% | 99.48% @t.028 | 91.06% @t.700 | 99.47% @t.436 | 99.24% @t.833 | 99.47% @t.011 | 99.53% @t.192 |
| 98% | 98.83% @t.882 | 82.05% @t.944 | 99.18% @t.509 | 98.00% @t.995 | 99.02% @t.212 | 99.00% @t.757 |

## Takeaways
- **FFAA alone is nearly sufficient frame-level.** Best fusion adds only ~0.05 pt at 90% real,
  because the ensemble uniquely catches only ~0.16% of fakes FFAA misses.
- **Combining pays at strict real-recall (≥98%):** equal `mean` → 99.18% vs FFAA 98.83% (~2,000
  more fake frames), and the blended threshold (~0.13–0.19 for weighted) is far stabler than
  FFAA-alone's twitchy near-zero τ.
- **Recommended operating points:** `weighted 0.2·ens+0.8·ffaa` (τ≈0.13) for 88–95% real;
  `mean` (τ≈0.51) for ≥98% real. The 100%/100% result reported earlier was *video-level*
  (per-clip mean aggregation), which is out of scope here (frame-level only).
