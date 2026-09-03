# PAAS_ensemble_v3

Standalone 5-detector face real/fake (liveness) ensemble. Extends PAAS_ensemble_v2 (FFAA + 9-class
A1/A2/A3) with the two CLIP detectors **GSD** (Exp 10) and **SeLop/LROR** (Exp 11), and fuses the
best subset found in Exp 13:

> **fused fake-score = mean( FFAA, A1_9c, A2_9c, GSD, SeLop )**

On axonlabs_data_1 (613,415 frames) this mean-of-5 reaches **AUC 0.9998**, fake-recall **99.97 %** at
a 90 % real-recall floor (see `docs/COMBINATION_FINDINGS_axon1.md`, Experiment 13). A3 is dropped from
the fusion (its saturated fake scores have no tail); it is still bundled and can be re-added via config.

Runs on the **global** `python3.12` / `transformers==4.37.2` interpreter (the FFAA / LLaVA-Mistral
stack) — not a venv.

## The five components

| component | what | source (Exp) | weights |
|-----------|------|--------------|---------|
| `ffaa`  | LLaVA-Mistral-7B MLLM + MIDS forgery scorer | 9  | `weights/ffaa_llava_mids/` |
| `A1_9c` | 9-class CLIP ensemble member (SVD)          | 7-8 | `weights/ensemble9/A1_svd_9c.pt` |
| `A2_9c` | 9-class CLIP ensemble member (SVD+GenD)      | 7-8 | `weights/ensemble9/A2_svdgend_9c.pt` |
| `gsd`   | Geometric Semantic Decoupling (dual CLIP)    | 10 | `weights/gsd/best_lastN_ep0_auc0.9930.pt` |
| `selop` | SeLop / LROR (frozen CLIP + low-rank orth.)   | 11 | `weights/selop/best.pt` |

All CLIP-based members share ONE backbone `base_models/clip-vit-large-patch14-336` (the vision
weights are identical across members). A3 (`weights/ensemble9/A3_midspp_9c.pt`) is also present.

## Layout

```
paas/                      unified pipeline (the single entry point)
  config.py                PaasConfig: which components + fusion + decision threshold
  fusion.py                mean / weighted over components; OPERATING_POINTS table
  pipeline.py              loads the needed detectors, scores, fuses, decides
  decision.py              fused score -> decision / match / forgery_type
  models/                  ffaa_model, ensemble9_model, gsd_model, selop_model  (uniform score_frames)
  data/face_filter.py      optional insightface real-quality filter
ffaa/  ensemble9/  gsd/  selop/     the four vendored model code trees
config/experiments/paas5_mean.json  the recommended config (default)
inference.py               CLI: score images
test_video_image_batch.py  multi-GPU batch tester over an image/video tree
weights/  base_models/     model checkpoints + shared CLIP backbone
# training (see "Training" below): gsd_train.py, selop_train.py, train_ensemble9.sh, run_finetuning.sh
```

## Inference

```bash
# single / several images (default config = mean of all 5)
python3.12 inference.py face.jpg
python3.12 inference.py a.jpg b.png --json
python3.12 inference.py --dir /path/to/folder

# a faster subset (no 7B MLLM): the two CLIP detectors only
python3.12 inference.py face.jpg --components gsd,selop

# a stricter operating point (real-98) — see paas/fusion.OPERATING_POINTS
python3.12 inference.py face.jpg --threshold 0.373
```

Programmatic:

```python
from paas.config import PaasConfig
from paas.pipeline import PaasPipeline
pipe = PaasPipeline(PaasConfig.from_file("config/experiments/paas5_mean.json"))
print(pipe.predict_images(["face.jpg"]))
```

Multi-GPU batch scoring of a real/fake folder tree (ground truth from the `real`/`fake` path part):

```bash
python3.12 test_video_image_batch.py \
    --input-dir /datasets/work/vLLM/data/axonlabs_data_1 \
    --out-dir runs/test --devices 0,1,2,3
```

## Operating points (mean-of-5 on axonlabs_data_1)

| real-recall floor | threshold | fake-recall |
|---|---|---|
| 80 % | 0.117 | 99.99 % |
| 90 % | **0.198** (default) | 99.97 % |
| 95 % | 0.263 | 99.89 % |
| 98 % | 0.373 | 99.77 % |
| 99 % | 0.449 | 99.58 % |

Set `decision.threshold` (config) or `--threshold` (CLI) to move the point.

## Training

Each detector trains independently; the ensemble is train-free (just mean-fusion).

- **GSD** — `python3.12 gsd_train.py --config configs/default.json` (embeds a fixed anchor U at the end).
- **SeLop** — `python3.12 selop_train.py --config selop_config.json`.
- **9-class A1/A2/A3** — `bash train_ensemble9.sh` (uses `ensemble9/mids9lib`, `ensemble9/scripts/`).
- **FFAA (LLaVA+MIDS)** — `bash run_finetuning.sh` (LoRA finetune + MIDS head).

Datasets: `train` = `testset/testset_mids/mids_first_half.json`, `val` = `.../mids_testset.json`
(3-class real/pad/deepfake via `get_label_all`; MAKEUP -> PAD).

## Notes

- `weights/` and `base_models/` are hardlinked from the source projects (same inode, no extra disk);
  they are real files, so the project is fully standalone.
- For an inference-only, optimized deployment with a FastAPI JSON service, see **PAAS_ensemble_v3_inf**.
