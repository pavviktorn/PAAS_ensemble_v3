# PAAS_ensemble_v2 — Improvement Plan toward 100% fake-recall / ≥95% real-recall (frame-level)

Consolidated from two literature sweeps (FAS / DFD, 2024–2026). Organized by **mechanism workstream**
(each fed by one or more papers), then a single **sequenced phase plan** with measurable gates.

---

## 0. The reframe that drives the priorities

Goal: **100% fake-recall AND ≥95% real-recall, frame-level.** From our own FILTERED heldout3 report,
the binding constraint is **real-recall, not fake-recall**:

| metric | current best member | target | gap |
|---|---|---|---|
| Fake-recall | ~99.7% (FFAA) / 94.8% (A3) | 100% | ~0.3–5 pts |
| **Real-recall** | **76–79%** (A3 76.3, A0 79.5) | **95%** | **~16–19 pts** |

~1 in 5 *clean, frontal, filtered* real frames is still flagged fake. So the plan is **dominated by
false-positive-on-real reduction**; fake-recall is nearly solved. The exact target metric is
**BPCER@APCER=0** (ISO/IEC 30107-3) = real-recall at the threshold where fake-recall = 100%.

Confirmed our stack IS the literature:
- A1/A2 SVD adaptation = **Effort** (Efficient Orthogonal Modeling, arXiv 2411.15633) — 0.19M params,
  orthogonal semantic vs forgery subspaces.
- A2/A3 GenD = **"Deepfake Detection that Generalizes Across Benchmarks"** (arXiv 2508.06248) —
  LayerNorm-only (0.03%) + hyperspherical L2 + metric learning, SOTA over 14 datasets.

Upgrades below are *successors* to these, not replacements.

---

## 1. Mechanism workstreams (all papers, deduped)

### WS-A — Semantic-shortcut / spurious-subspace removal  (reduces real-FP)  ★ top leverage
The dominant low-rank directions of CLIP features encode forgery-IRRELEVANT priors (capture
conditions, identity, dataset look). Real video frames with unusual capture get pushed up the fake
axis ("semantic fallback"). Remove that subspace; learn forgery in its orthogonal complement.
Three cost points, same idea — use as a progression:
- **Effort** (arXiv 2411.15633) — *have it* (A1/A2). Fixed SVD split, adapt residual only.
- **GSD — Geometric Semantic Decoupling** (When Detectors Forget Forensics, arXiv 2603.09242) —
  **parameter-free**, per-batch Householder/QR projection of learnable features onto the semantic
  null space. Dynamic (batch stats) → catches nuisance shortcuts Effort's static split misses.
  94.4% avg AUC, +3% unseen. **Drop-in, ~0 new params.** → do first.
- **SeLop — Low-rank Orthogonal Subspace Intervention** (arXiv 2601.11915) — trainable orthogonal
  complement, 0.39M params, causal framing. → if GSD's free projection under-delivers, add this.
Target files: `ensemble9/mids9lib/svd.py`, `model.py` (forward feature path).

### WS-B — Style / nuisance decoupling  (reduces real-FP)
Explicitly model capture nuisance and factor it OUT of the spoof decision (compute fake-score on
content-only features).
- **InstructFLIP content/style decoupling + cue generator** (arXiv 2507.12060) — a style branch
  models illumination/camera/environment; drives TPR@FPR=1% from 54%→65%. 1:1 with our real-FP.
- **Disentangle liveness-irrelevant factors + finer domain partition** (arXiv 2407.08243) —
  FAS-specific disentanglement of capture/style/quality from liveness.
Action: add a style/nuisance head or style-conditioned normalization to the ensemble.
Target files: `ensemble9/mids9lib/model.py`, `gend.py`.

### WS-C — Real-class strengthening  (reduces real-FP, data + objective side)
- **Real-only spoof-prompt learning** (Learning Unknown Spoof Prompts Using Only Real Face Images,
  arXiv 2505.03611) — train on ONLY reals + learned CLIP spoof-prompts + one-class real
  discrimination → high-confidence reals; 28% avg error reduction. Use as a **4th ensemble member
  that vetoes false fakes** (a real-confidence gate).
- **Hard-negative mining on our actual FP reals + richer bona-fide prompts** — mine the `miss/`
  reals flagged fake, add as hard negatives in the next fine-tune; enrich the "real" text-prompt
  set with bona-fide synonyms (authentic/genuine/unaltered/original). Lowest effort.
Target files: `train_ensemble9.sh` (warm-start data), `ensemble9/configs/*.yaml` (prompt set),
new member wired through `paas/fusion.py`.

### WS-D — Post-hoc decision layer  (pins fake-recall, recovers real-recall, NO retraining)  ★ fastest
- **Test-Time Domain Generalization for FAS** (arXiv 2403.19334) — at inference, project each test
  frame into the *training* style space; needs only stored training style stats. Drop-in transform.
- **Neyman–Pearson thresholding** (Scott NP design; survey PMC5804623) — the correct tool for the
  literal goal: guarantees fake-recall ≥ 1−α *on the population w.h.p.* while minimizing real-FP.
  Strictly better than "threshold at 100% val fake-recall" (which overfits the val tail).
- **Conformal prediction + abstention band** (Visual Intelligence 2025, s44267-025-00100-2; ICCVW
  2025 "Is It Certainly a Deepfake?"; Architecture-Adaptive Uncertainty Fusion, arXiv 2606.06666) —
  flag the overlap zone "uncertain → review" instead of forcing a wrong class.
Target files: `paas/decision.py`, `paas/fusion.py`, `scripts/combine_eval.py`, `train/train_combiner.py`,
inference transform in `paas/pipeline.py`.

### WS-E — Fake-recall / detail sensitivity  (lower priority given the gap)
Helps hard *fakes* (PAD/screen, subtle blends), not the binding real gap — schedule only if fake-recall
slips below 100% after the real-FP work.
- **I-FAS: Globally Aware Connector + Lopsided LM Loss** (arXiv 2501.01720) — GAC fuses multi-level
  CLIP features (not just layer −2) for moiré/blur/screen; lopsided loss prioritizes the judgment over
  the caption → sharper FFAA fake-score. Apply in FFAA Step-1 LoRA (`ffaa/llava/train/train_mem.py`,
  `mm_vision_select_layer`).
- **Deepfake Forensics Adapter (dual-stream)** (arXiv 2603.01450) — frozen CLIP + global
  attention-bias adapter + local landmark-region stream (eyes/mouth/nose) for blending artifacts.
- **GenD v3 recipe refresh** (arXiv 2508.06248v3) — confirm A2/A3 use current LayerNorm +
  hyperspherical + metric-learning recipe (cheap config/loss delta).

### Dismissed
- Facial Component Guidance (CVPR 2025) and temporal methods — video-based, out of scope (frame-level).
- Multi-Artifact Subspace Decomposition (arXiv 2601.01041) — CNN (Xception/EfficientNet), not CLIP;
  WS-A subsumes the useful part for a CLIP stack.
- CCPE (arXiv 2504.04470) — LLM composite prompts; marginal text-side gain, lower priority than WS-C.

---

## 2. Sequenced phase plan (each phase gated by a number)

Assets in place: 30,219-image `/datasets/work/vLLM/temp/testset`, `mids9_val.json`, warm-start
`train_ensemble9.sh`, `mids9lib/{svd,gend,losses,model}.py`, `paas/{fusion,decision,pipeline}.py`,
`scripts/combine_eval.py`, and the `miss/`-mining tester.

**Phase 0 — Diagnose before building (½–1 day, read-only, no GPU).**
Run the tester on `testset` per member (A1/A2/A3/FFAA); mine `miss/`. Quantify: real-recall per real
subset (confirm R_12/R_15 wall); the visual property of FP reals (cluster CLIP embeddings of FP vs TN
reals — low-light? compression? pose?); which model drives each FP; frame-level real-vs-fake score
overlap.
→ **Gate:** one-page failure profile. Capture-nuisance cluster ⇒ bet on WS-A + WS-D. Scattered hard
cases ⇒ bet on WS-C + WS-D.

**Phase 1 — Post-hoc layer, zero retraining (1–2 days). [WS-D]  Target: real-recall 76→~85%, fake-recall pinned.**
- 1a. Temperature-scale each model's fake-score on a calibration split of `mids9_val`; refit fusion.
- 1b. Replace single τ with NP-controlled threshold (fake-recall ≥99.5–100% w.h.p.) + conformal abstain band.
- 1c. Add Test-Time style projection as an optional `pipeline.py` inference transform.
→ **Gate:** BPCER@APCER=0 on a held-out half of testset vs Phase-0 baseline.

**Phase 2 — Semantic-shortcut removal on the ensemble (3–5 days). [WS-A]  Target: real-recall →90%+.**
- Add parameter-free GSD projection in A1/A2/A3 forward path (`model.py`/`svd.py`).
- Warm-start retrain via `train_ensemble9.sh` (already wired: `--resume`, GPU list, OMP cap).
- If short: add trainable SeLop orthogonal complement.
→ **Gate:** per-member real-recall lift, fake-recall held ≥99%.

**Phase 3 — Real-class strengthening (1 week). [WS-C]  Target: ≥95% real.**
- Train real-only spoof-prompt member on filtered reals; add as 4th member (false-fake veto) in `paas/fusion.py`.
- Hard-negative mine Phase-0 FP reals into the warm-start set; enrich bona-fide prompt set.
→ **Gate:** combined real-recall ≥95% at fake-recall ≥99.5% on testset.

**Phase 4 — Only if still short. [WS-B / WS-E]**
- WS-B style-decoupling head if Phase-0 showed a capture-nuisance cluster GSD didn't fully clear.
- WS-E (FFAA GAC + lopsided, dual-stream adapter, GenD-v3) only if fake-recall dropped below 100%.

**Throughout:** primary metric = **BPCER@APCER=0** (real-recall at 100% fake-recall); also track HTER and
per-subset real-recall (sACC across the 46 testset subsets).

---

## 3. Fastest credible path
Phase 0 → Phase 1 (days, no GPU retraining, does not disturb the running ensemble training) literally
satisfies "100% fake-recall at the best achievable real-recall." The durable lift to 95% real comes from
Phase 2 (GSD) + Phase 3 (real-only member).
