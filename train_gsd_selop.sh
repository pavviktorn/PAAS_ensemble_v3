#!/usr/bin/env bash
# Train BOTH CLIP detectors concurrently, one per GPU pair, in a single run:
#     GSD   -> GPUs 0,1   (DataParallel,  gsd_train.py)
#     SeLop -> GPUs 2,3   (DDP via torch.distributed.run, selop_train.py)
#
# Same data for both:
#     train = /datasets/work/vLLM/temp/mids_plus/runs/real/mids9_full/mids9.json
#     eval  = /datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json
#
# Runs on the GLOBAL python3.12. Override anything via env, e.g.:
#     GSD_GPUS=0,1 SELOP_GPUS=2,3 TR=... VA=... bash train_gsd_selop.sh
#     LIMIT=64 bash train_gsd_selop.sh            # tiny smoke run (caps train/val samples)
#     SELOP_SCRATCH=1 bash train_gsd_selop.sh     # SeLop from scratch (ignore config init_from)
set -e
PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT"
PY=python3.12

# ---- data + backbone ----
TR="${TR:-/datasets/work/vLLM/temp/mids_plus/runs/real/mids9_full/mids9.json}"
VA="${VA:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"
CLIP="$PROOT/base_models/clip-vit-large-patch14-336"
GSD_GPUS="${GSD_GPUS:-0,1}"
SELOP_GPUS="${SELOP_GPUS:-2,3}"
# warm-start (init) from the current best checkpoints. GSD: --init-from (weights only, fresh optimizer);
# SeLop: via its config "init_from" (already weights/selop/best.pt). Set GSD_SCRATCH=1 for GSD from scratch.
GSD_INIT="${GSD_INIT:-$PROOT/weights/gsd/best_lastN_ep0_auc0.9930.pt}"
OUT="$PROOT/runs"
GSD_OUT="$OUT/gsd_train_full"; SELOP_OUT="$OUT/selop_train_full"
mkdir -p "$GSD_OUT" "$SELOP_OUT"

for f in "$TR" "$VA"; do [ -f "$f" ] || { echo "missing dataset: $f" >&2; exit 1; }; done

# ---- CPU cap (~50% of 96 cores across both jobs): GSD cpu_fraction 0.25 + SeLop 2x12 threads ----
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROOT:${PYTHONPATH:-}"

# optional smoke-test limits
GSD_LIMIT=(); SELOP_LIMIT=()
if [ -n "${LIMIT:-}" ]; then
  GSD_LIMIT=(--limit "$LIMIT")
  SELOP_LIMIT=(--limit_train "$LIMIT" --limit_val "$LIMIT")
  echo "[train] SMOKE: capping train/val to $LIMIT samples"
fi

echo "==================================================================="
echo "[train] GSD   -> GPUs $GSD_GPUS   out=$GSD_OUT"
echo "[train] SeLop -> GPUs $SELOP_GPUS out=$SELOP_OUT"
echo "[train] train=$TR"
echo "[train] eval =$VA"
echo "==================================================================="

# ---------- GSD (DataParallel over GSD_GPUS; the process reads gpus=... , no CVD remap) ----------
GSD_INIT_ARG=()
if [ "${GSD_SCRATCH:-0}" != "1" ] && [ -f "$GSD_INIT" ]; then
  GSD_INIT_ARG=(--init-from "$GSD_INIT")
  echo "[train] GSD warm-start (init) from $GSD_INIT"
else
  echo "[train] GSD from scratch (GSD_SCRATCH=${GSD_SCRATCH:-0}, init exists=$([ -f "$GSD_INIT" ] && echo yes || echo no))"
fi
$PY gsd_train.py --config configs/default.json "${GSD_LIMIT[@]}" "${GSD_INIT_ARG[@]}" \
    --set train_data="$TR" val_data="$VA" clip_path="$CLIP" gpus="$GSD_GPUS" \
          cpu_fraction=0.25 output_dir="$GSD_OUT" \
    > "$GSD_OUT/train.stdout.log" 2>&1 &
GSD_PID=$!
echo "[train] GSD launched (pid $GSD_PID) -> tail -f $GSD_OUT/train.stdout.log"

# ---------- SeLop (DDP): selop_train reads train/val from its config, so derive a run-config ----------
SELOP_CFG="$SELOP_OUT/run_config.json"
"$PY" - "$PROOT/selop_config.json" "$TR" "$VA" "$CLIP" "$SELOP_OUT" "$SELOP_CFG" "${SELOP_SCRATCH:-0}" <<'PYEOF'
import json, sys
src, tr, va, clip, out, dst, scratch = sys.argv[1:8]
c = json.load(open(src))
c["train_data"], c["val_data"], c["clip_path"], c["out_dir"] = tr, va, clip, out
if scratch == "1":
    c.pop("init_from", None)          # train from scratch instead of warm-starting from best.pt
json.dump(c, open(dst, "w"), indent=2)
print(f"[selop] run-config -> {dst} (init_from={c.get('init_from')})")
PYEOF
NPROC=$(echo "$SELOP_GPUS" | awk -F, '{print NF}')
CUDA_VISIBLE_DEVICES="$SELOP_GPUS" $PY -m torch.distributed.run --nproc_per_node="$NPROC" \
    --master_port="${MASTER_PORT:-29534}" \
    selop_train.py --config "$SELOP_CFG" "${SELOP_LIMIT[@]}" \
    > "$SELOP_OUT/train.stdout.log" 2>&1 &
SELOP_PID=$!
echo "[train] SeLop launched (pid $SELOP_PID, nproc=$NPROC) -> tail -f $SELOP_OUT/train.stdout.log"

# ---------- wait for both, report ----------
GSD_RC=0; SELOP_RC=0
wait "$GSD_PID"   || GSD_RC=$?
wait "$SELOP_PID" || SELOP_RC=$?
echo "==================================================================="
echo "[train] DONE. GSD exit=$GSD_RC (best -> $GSD_OUT) | SeLop exit=$SELOP_RC (best -> $SELOP_OUT)"
echo "[train] promote to weights/ when happy, e.g.:"
echo "        cp $GSD_OUT/best*.pt   weights/gsd/"
echo "        cp $SELOP_OUT/best*.pt weights/selop/"
[ "$GSD_RC" -eq 0 ] && [ "$SELOP_RC" -eq 0 ]
