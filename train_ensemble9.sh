#!/usr/bin/env bash
# Fine-tune the 3 MLLM-free 9-class ensemble members on the new mids9.json, validating on
# mids9_val.json (built from heldout3 with FILTERED reals). Runs on the GLOBAL python3.12.
#
#   A1 = +SVD only            A2 = +SVD+GenD            A3 = full MIDS++ (SVD+GenD+artifact)
#
# One member per GPU. Override GPUs / data / epochs via env vars. After training, copy each
# member's best.pt into weights/ensemble9/ and point config/ensemble9.json at them.
set -e
PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT/ensemble9"
export PYTHONPATH="$PROOT/ensemble9" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3.12

# CPU thread cap. The 3 members train as 3 separate processes, each defaulting its OMP/MKL pool to
# the full core count -> 3 x 96 threads on 96 cores = every core pinned at 100%. Cap per-process
# threads instead: total busy cores ~= OMP_NUM_THREADS x 3 members. Default 16 x 3 = ~48 cores (~50%
# of 96). Set OMP_NUM_THREADS=96 for full speed, =8 for ~25%.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
CLIP="$PROOT/base_models/clip-vit-large-patch14-336"
T5="$PROOT/base_models/t5-base"
TR="${TR:-/datasets/work/vLLM/temp/mids_plus/runs/real/mids9_full/mids9.json}"
VA="${VA:-/datasets/work/vLLM/temp/mids_plus/runs/real/mids9/mids9_val.json}"
OUT="$PROOT/runs/ensemble9_train_full"; mkdir -p "$OUT"

# GPU assignment. One member per GPU. Override the whole list with GPUS=1,2,3 (the 3 members map to
# the listed devices in order, wrapping if you give fewer than 3), or pin individually via G1/G2/G3.
GPUS="${GPUS:-1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NG="${#GPU_ARR[@]}"
[ "$NG" -ge 1 ] || { echo "no GPUs in GPUS=$GPUS" >&2; exit 1; }
G1="${G1:-${GPU_ARR[0]}}"
G2="${G2:-${GPU_ARR[$((1 % NG))]}}"
G3="${G3:-${GPU_ARR[$((2 % NG))]}}"
echo "[train_ensemble9] GPUs: A1->$G1  A2->$G2  A3->$G3"

# Warm-start: fine-tune the PREVIOUS A1/A2/A3 checkpoints instead of training from scratch.
# Trainable weights (SVD residuals, GenD LayerNorms, artifact head, classifier) are loaded via
# --resume (strict=False, with the CLIP key-remap fix). Set SCRATCH=1 to disable and train fresh,
# or override any member's init path via INIT_A1/INIT_A2/INIT_A3.
W="$PROOT/weights/ensemble9"
INIT_A1="${INIT_A1:-$W/A1_svd_9c.pt}"
INIT_A2="${INIT_A2:-$W/A2_svdgend_9c.pt}"
INIT_A3="${INIT_A3:-$W/A3_midspp_9c.pt}"
if [ "${SCRATCH:-0}" = "1" ]; then
  R1=(); R2=(); R3=(); echo "[train_ensemble9] SCRATCH=1 -> training from scratch"
else
  R1=(--resume "$INIT_A1"); R2=(--resume "$INIT_A2"); R3=(--resume "$INIT_A3")
  echo "[train_ensemble9] warm-start: A1<-$INIT_A1  A2<-$INIT_A2  A3<-$INIT_A3"
fi
COMMON="train_data_path=$TR val_data_path=$VA image_model_path=$CLIP text_model_path=$T5 \
num_classes=9 samples_per_image=3 skip_missing_images=true epochs=${EPOCHS:-200} \
batch_size=${BS:-24} val_batch_size=16 amp_dtype=bf16 num_workers=8 save_every_steps=2000"

CUDA_VISIBLE_DEVICES=$G1 $PY -m mids9lib.train --config configs/mids_pp.yaml "${R1[@]}" \
  --set $COMMON artifact_enabled=false alignment_weight=0 uniformity_weight=0 query_orth_weight=0 \
  tune_layer_norm=false output_dir="$OUT/A1_svd" > "$OUT/A1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 $PY -m mids9lib.train --config configs/mids_pp.yaml "${R2[@]}" \
  --set $COMMON artifact_enabled=false query_orth_weight=0 output_dir="$OUT/A2_svdgend" > "$OUT/A2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G3 $PY -m mids9lib.train --config configs/mids_pp.yaml "${R3[@]}" \
  --set $COMMON output_dir="$OUT/A3_midspp" > "$OUT/A3.log" 2>&1 &
wait
echo "done -> $OUT/{A1_svd,A2_svdgend,A3_midspp}/best.pt"
echo "deploy: cp those best.pt into $PROOT/weights/ensemble9/ and update config/ensemble9.json members[].checkpoint"
