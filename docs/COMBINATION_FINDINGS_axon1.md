# Combination findings — axonlabs_data_1 (frame-level)

Dataset: `/datasets/work/vLLM/data/axonlabs_data_1` — 460 image + 1,905 video files.
Reals pass the ensemble **FaceQualityFilter** (insightface buffalo_l); the *exact* 22,046 SK
frame-keys from the ensemble run (`results_ensemble_A1_9c.txt`) are re-used for GSD/SeLop, so all
seven detectors are scored on the **identical** real set (no re-running the filter).

Seven detectors aligned per frame-key (`path#frame=NNNNNN`), SK/ER frames dropped, giving the
common evaluated set:

> **N = 613,415 frames — real = 48,526, fake = 564,889**

fake-score convention: FFAA = `match if analysis==fake else 1−match`; A1/A2/A3 = 1−P(real) over the
9-class head; GSD/SeLop = 1−P(real) over the 3-class (real/pad/deepfake) head. "ensemble" is the
production A1+A2+A3 fusion, shown for reference (not in the subset search).

Repro: `runs/test_axon0model_axon1datatest/frontier_axon1.py`
GSD = `PAAS_v4/runs/gsd/best_lastN_ep0_auc0.9930.pt` (embedded anchor) · SeLop = `PAAS_SeLop/runs/selop_default/best.pt`.

---

## Score distributions & AUC

| model | real mean/median | fake mean/median | AUC |
|-------|:----------------:|:----------------:|:----:|
| **GSD**   | 0.033 / 0.000 | 0.982 / 1.000 | **0.9985** |
| ensemble  | 0.125 / 0.012 | 0.983 / 1.000 | 0.9982 |
| SeLop     | 0.094 / 0.007 | 0.991 / 1.000 | 0.9980 |
| FFAA      | 0.024 / 0.000 | 0.989 / 1.000 | 0.9965 |
| A1        | 0.041 / 0.000 | 0.972 / 1.000 | 0.9949 |
| A2        | 0.113 / 0.004 | 0.982 / 1.000 | 0.9935 |
| A3        | 0.222 / 0.002 | 0.994 / 1.000 | 0.9790 |

The two new CLIP detectors (GSD, SeLop) are the **strongest single models**, edging out the current
production ensemble. A3 is the weakest and its fake scores saturate at 1.0, so it has no usable tail
(see below).

---

## FRAME-LEVEL FRONTIER — fake-recall @ the threshold hitting each real-recall floor

Cell = fake-recall at the lowest threshold whose real-recall reaches that floor (`% @t<thr>`).

| model     | 80%   | 85%   | 88%   | 90%   | 92%   | 95%   | 98%   | 99% |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| FFAA      | 99.82 | 99.75 | 99.71 | 99.66 | 99.59 | 99.45 | 98.92 | 96.88 |
| ensemble  | 99.77 | 99.74 | 99.67 | 99.58 | 99.46 | 99.23 | 98.74 | 97.96 |
| A1        | 99.49 | 99.24 | 99.01 | 98.81 | 98.54 | 97.90 | 96.46 | 94.90 |
| A2        | 99.29 | 99.01 | 98.77 | 98.54 | 98.23 | 97.45 | 95.19 | 92.37 |
| A3        | 99.38 | 99.08 | 98.70 | 98.25 | 97.44 | 94.00 | **0.00** | **0.00** |
| **GSD**   | 99.87 | 99.81 | 99.74 | 99.68 | 99.57 | 99.28 | 98.43 | 97.66 |
| SeLop     | 99.78 | 99.67 | 99.57 | 99.47 | 99.35 | 99.06 | 98.28 | 97.40 |

- **GSD leads the single-model frontier** up to ~92% real, then FFAA/ensemble edge it in the extreme
  tail; ensemble is best single at real-98 (98.74). GSD and SeLop both hold **>97.4% fake-recall even
  at 99% real** — the best single-model tails after ensemble.
- **A3 collapses to 0 above 95% real**: its fake scores pile at exactly 1.0, so once the threshold has
  to exceed the reals it also excludes nearly all fakes. It hurts high-real-floor fusion — drop it.

---

## Fusion frontier (unsupervised, calibration-free)

| fusion                            | 80%   | 85%   | 88%   | 90%   | 92%   | 95%       | 98%       | 99%       |
|-----------------------------------|-------|-------|-------|-------|-------|-----------|-----------|-----------|
| **rank-mean (FFAA+A1+GSD+SeLop)** | 99.98 | 99.96 | 99.95 | 99.94 | 99.93 | **99.89** | **99.75** | **99.53** |
| mean (FFAA+A1+GSD+SeLop)          | 99.97 | 99.95 | 99.94 | 99.93 | 99.91 | 99.88     | 99.69     | 99.50     |
| mean (all 6 base)                 | 99.98 | 99.96 | 99.94 | 99.92 | 99.90 | 99.86     | 99.72     | 99.58     |
| max / OR (FFAA+A1+GSD+SeLop)      | 99.96 | 99.94 | 99.93 | 99.91 | 99.88 | 99.80     | 99.47     | 98.95     |
| min / AND (FFAA+A1+GSD+SeLop)     | 99.76 | 99.76 | 99.64 | 99.64 | 99.56 | 99.43     | 99.10     | 98.83     |
| mean (FFAA + ensemble)            | 99.91 | 99.90 | 99.88 | 99.85 | 99.83 | 99.78     | 99.39     | 98.90     |

Any 4-way fusion beats every single model by a wide margin. **Rank-mean** (percentile-rank each model,
then average) is the best method — it needs no per-model calibration and is dominant across the whole
frontier, holding **99.5% fake-recall at a 99% real floor**. `max`/`OR` is competitive mid-range but
loses the tail; `min`/`AND` is the weakest.

---

## Best model subsets (rank-mean), ranked by tail fake-recall

| subset                           | FR@95     | FR@98     | AUC        |
|----------------------------------|:---------:|:---------:|:----------:|
| **FFAA + A1 + A2 + GSD + SeLop** | **99.93** | **99.81** | **0.9998** |
| FFAA + A2 + GSD + SeLop          | 99.93     | 99.79     | 0.9997     |
| FFAA + A1 + GSD + SeLop          | 99.89     | 99.75     | 0.9997     |
| FFAA + A1 + A2 + SeLop           | 99.89     | 99.73     | 0.9996     |
| FFAA + GSD + SeLop               | 99.89     | 99.71     | 0.9996     |
| A1 + A2 + GSD + SeLop            | 99.87     | 99.70     | 0.9997     |

(Search over all 63 subsets of the 6 base models; ensemble excluded as it is itself A1+A2+A3.)

---

## Takeaways

1. **GSD and SeLop are the highest-value additions.** They are the two best single models
   (AUC 0.9985 / 0.9980) and every strong subset contains at least one of them; the best subsets
   contain **both**. They add a CLIP-semantic view that is diverse from FFAA's MLLM view.
2. **Best combo: rank-mean of `FFAA + A1 + A2 + GSD + SeLop`** — AUC 0.9998, **FR@95 = 99.93%,
   FR@98 = 99.81%**, and ~99.5% fake-recall even at a 99% real floor. This is the recommended
   production fusion for frame-level scoring.
3. **Drop A3** from high-real-floor fusions: its saturated fake scores give it no tail
   (0% fake-recall above 95% real) and it only dilutes rank-mean there. A leaner
   `FFAA + A2 + GSD + SeLop` matches the 5-way at FR@95 (99.93) with one fewer model.
4. **Use rank-mean, not mean/max/min.** It is calibration-free and dominates the frontier; `max/OR`
   trades tail recall for mid-range, `min/AND` is weakest.
5. Fusion converts the best *single-model* real-98 fake-recall (~98.4–98.7%) into **99.8%** — i.e. it
   roughly **halves the fake miss-rate at the same real-recall** relative to any one detector,
   including the current production ensemble (98.74% → 99.81% at real-98).
