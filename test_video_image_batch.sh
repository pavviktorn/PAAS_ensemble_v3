#!/usr/bin/env bash
# Run test_video_image_batch.py once -- it now orchestrates multi-GPU internally (one worker process
# per GPU, file-sharded, with real cross-file batching) and merges the shard outputs itself. This
# wrapper just sets the env knobs and runs the offline combination frontier afterwards.
# Global python3.12 / tf==4.37.2.
#
# Usage:
#   DATA=/path/to/data bash test_video_image_batch.sh
#   DATA=/data OUT=runs/axon GPUS=0,1,2,3 CONFIG=config/experiments/weighted_ffaa.json \
#       FRAME_STRIDE=10 FUSION= THRESHOLD= FILTER_REAL=0 bash test_video_image_batch.sh
#
# NOTE: if a training run is using the GPUs, this will contend heavily -- run it when the box is
# free, or it may get OOM/governor-killed.
set -u

PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT"

# ---- config (override via env) ----
CONFIG="${CONFIG:-config/experiments/weighted_ffaa.json}"
DATA="${DATA:-/datasets/work/vLLM/data/axonlabs_data_1}"
OUT="${OUT:-runs/test_axon0model_axon1datatest}"
GPUS="${GPUS:-0,1,2,3}"          # passed straight to --devices
FRAME_STRIDE="${FRAME_STRIDE:-1}"
FFAA_BATCH="${FFAA_BATCH:-64}"
ENS_BATCH="${ENS_BATCH:-64}"
FLUSH_SIZE="${FLUSH_SIZE:-256}"
FILTER_REAL="${FILTER_REAL:-1}"
LIMIT="${LIMIT:-0}"
FUSION="${FUSION:-}"            # optional: override fusion method (e.g. ensemble_only / mean)
THRESHOLD="${THRESHOLD:-}"      # optional: override decision threshold
FFAA_CACHE="${FFAA_CACHE:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"   # optional: pre-generated FFAA answers JSON (skips LLaVA gen on hits;
                               # misses fall back to normal generation). e.g.
                               # /datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json
# CPU thread cap. The script itself caps torch threads to cores/num-GPUs; OMP_NUM_THREADS still
# bounds the BLAS pools shared across the worker processes. Default 12; set 96 for full speed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
NICE="${NICE:-0}"

mkdir -p "$OUT"

extra=()
[ -n "$FUSION" ]     && extra+=(--fusion "$FUSION")
[ -n "$THRESHOLD" ]  && extra+=(--threshold "$THRESHOLD")
[ -n "$FFAA_CACHE" ] && extra+=(--ffaa-cache "$FFAA_CACHE")

echo "=== test_video_image_batch.py on GPUs [$GPUS] | config=$CONFIG | data=$DATA | out=$OUT ==="
nice -n "$NICE" python3.12 test_video_image_batch.py \
    --config "$CONFIG" \
    --input-dir "$DATA" \
    --out-dir "$OUT" \
    --devices "$GPUS" \
    --frame-stride "$FRAME_STRIDE" --ffaa-batch "$FFAA_BATCH" --ens-batch "$ENS_BATCH" \
    --flush-size "$FLUSH_SIZE" \
    --filter-real "$FILTER_REAL" --limit "$LIMIT" \
    "${extra[@]}"
fail=$?

if [ -f "$OUT/results_ensemble.txt" ] && [ -f "$OUT/results_ffaa.txt" ]; then
  echo "=== offline combination frontier (all fusions, no model re-run) ==="
  python3.12 scripts/combine_eval.py --ensemble "$OUT/results_ensemble.txt" --ffaa "$OUT/results_ffaa.txt" || true
fi

echo "=== done (misses in $OUT/miss/) ==="
exit "$fail"
