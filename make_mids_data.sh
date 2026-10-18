#!/usr/bin/env bash
# Generate the MLLM-referred MIDS json for an image tree (e.g. the new testset) using the FFAA MLLM
# (LLaVA-Mistral-7B). Recurses --input_dir, asks "Is it real or fake? Why?" per image, and records
# the MLLM's analysis + answer together with the ground-truth label derived from the path
# (real / fake-pad / fake-deepfake via get_label_all). Sharded one process per GPU, then merged into
# one combined json. Runs on the GLOBAL python3.12 / transformers==4.37.2.
#
# Usage:
#   bash make_mids_data.sh                                   # testset, FFAA bundled MLLM, GPUs 1,2,3
#   DATA=/path/to/tree OUT=runs/x/mids_x.json GPUS=0,1,2,3 \
#     MODEL=runs/mllm_merged BATCH=96 bash make_mids_data.sh
#
# The tree must use real/ and fake/{pad,deepfake}/ path components (the testset layout).
set -u

PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT/ffaa"
export PYTHONPATH="$PROOT/ffaa:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PAAS_CLIP="$PROOT/base_models/clip-vit-large-patch14-336"
export PAAS_T5="$PROOT/base_models/t5-base"

# ---- config (override via env) ----
MODEL="${MODEL:-$PROOT/weights/ffaa_llava_mids}"        # FFAA MLLM dir; set MODEL=$PROOT/runs/mllm_merged for a freshly-trained one
DATA="${DATA:-/datasets/work/vLLM/temp/testset}"
OUT="${OUT:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"
PROMPTS="${PROMPTS:-$PROOT/ffaa/playground/prompts.txt}"
GPUS="${GPUS:-0,1,2,3}"
BATCH="${BATCH:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OMP="${OMP_NUM_THREADS:-6}"
NICE="${NICE:-0}"
# Bad-format archives. The batch pass copies images whose MLLM output won't parse into ERR_PATH
# (by label); the one-by-one pass then re-processes ERR_PATH and copies its own still-failures into
# ERR_PATH2. Kept under the testset so a run stays self-contained.
ERR_PATH="${ERR_PATH:-/datasets/work/vLLM/temp/testset/error}"
ERR_PATH2="${ERR_PATH2:-/datasets/work/vLLM/temp/testset/error2}"
RECOVER="${RECOVER:-1}"          # 1 = run the one-by-one recovery pass over ERR_PATH afterwards
REPEAT="${REPEAT:-1}"            # one-by-one --repeat (augmentation passes per recovered image)
PROC_PER_GPU="${PROC_PER_GPU:-4}"  # recovery (one-by-one) processes per GPU; total shards = N * this

IFS=',' read -ra G <<< "$GPUS"; N="${#G[@]}"
[ "$N" -ge 1 ] || { echo "no GPUs in GPUS=$GPUS" >&2; exit 1; }
[ -e "$MODEL" ]  || { echo "MODEL not found: $MODEL" >&2; exit 1; }
[ -d "$DATA" ]   || { echo "DATA dir not found: $DATA" >&2; exit 1; }
[ -f "$PROMPTS" ]|| { echo "prompt list not found: $PROMPTS" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
dir="$(dirname "$OUT")"; base="$(basename "${OUT%.json}")"   # shard files: <dir>/<base>_<i>.json
rbase="${base}_recover"                                      # one-by-one recovery shard files

# Start the error archive clean so its post-run contents reflect THIS run only.
rm -rf "$ERR_PATH"; mkdir -p "$ERR_PATH"

echo "=== [1/3] FFAA MLLM -> MIDS json (batch) | model=$MODEL | data=$DATA | shards=$N on GPUs [$GPUS] | out=$OUT ==="
echo "          bad-format images archived to $ERR_PATH"
pids=()
for i in "${!G[@]}"; do
  gpu="${G[$i]}"
  echo "  shard $i/$N -> GPU $gpu (log: $dir/${base}_$i.log)"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" \
    nice -n "$NICE" python3.12 make_mids_dataset_from_folder_batch.py \
      --model_path "$MODEL" \
      --input_dir "$DATA" --image_root / \
      --output_json "$OUT" \
      --prompt_list "$PROMPTS" \
      --err_path "$ERR_PATH" \
      --which_part "$i" --n_divided "$N" \
      --device 0 --batch_size "$BATCH" --num_workers "$OMP" \
      --max_new_tokens "$MAX_NEW_TOKENS" &
  pids+=("$!")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "  shard $i done"; else echo "  shard $i FAILED (see $dir/${base}_$i.log)"; fail=1; fi
done

# ---- [2/3] one-by-one recovery of bad-format images, if any ----
NR=0   # number of recovery shards actually launched (0 if recovery is skipped)
nerr=$(find "$ERR_PATH" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
if [ "$RECOVER" = "1" ] && [ "$nerr" -gt 0 ]; then
  NR=$(( N * PROC_PER_GPU ))       # total recovery shards = 5 processes per GPU
  echo "=== [2/3] recovering $nerr bad-format image(s) one-by-one over $ERR_PATH | $NR shards ($PROC_PER_GPU/GPU x $N GPUs) | still-failures -> $ERR_PATH2 ==="
  rm -rf "$ERR_PATH2"; mkdir -p "$ERR_PATH2"
  rpids=()
  for j in $(seq 0 $((NR - 1))); do
    gpu="${G[$((j % N))]}"         # round-robin: shards j, j+N, j+2N ... share a GPU
    echo "  recover shard $j/$NR -> GPU $gpu (log: $dir/${rbase}_$j.log)"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" \
      nice -n "$NICE" python3.12 make_mids_dataset_from_folder_onebyone.py \
        --model_path "$MODEL" \
        --input_dir "$ERR_PATH" --image_root / \
        --output_json "$dir/$rbase.json" \
        --prompt_list "$PROMPTS" \
        --err_path "$ERR_PATH2" \
        --which_part "$j" --n_divided "$NR" \
        --device 0 --batch_size "$BATCH" --num_workers "$OMP" \
        --repeat "$REPEAT" --max_new_tokens "$MAX_NEW_TOKENS" &
    rpids+=("$!")
  done
  for j in "${!rpids[@]}"; do
    if wait "${rpids[$j]}"; then echo "  recover shard $j done"; else echo "  recover shard $j FAILED (see $dir/${rbase}_$j.log)"; fail=1; fi
  done
  nstill=$(find "$ERR_PATH2" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)
  echo "  recovery done; $nstill image(s) still unparseable (left in $ERR_PATH2)"
else
  echo "=== [2/3] no bad-format images to recover (RECOVER=$RECOVER, nerr=$nerr) ==="
fi

echo "=== [3/3] merging batch + recovery shards -> $OUT ==="
python3.12 - "$OUT" "$N" "$NR" "$dir" "$base" "$rbase" <<'PY'
import json, os, sys
from collections import Counter
out, n, nr, dir_, base, rbase = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
allr, seen = [], set()
def add(p, tag):
    if not os.path.exists(p):
        return
    d = json.load(open(p)); kept = 0
    for r in d:
        k = r.get("image")
        if k in seen:                 # de-dup: a recovered image may also linger in a batch shard
            continue
        seen.add(k); allr.append(r); kept += 1
    print(f"  + [{tag}] {os.path.basename(p)}: {kept}/{len(d)} records")
for i in range(n):
    add(os.path.join(dir_, f"{base}_{i}.json"), "batch")
for i in range(nr):
    add(os.path.join(dir_, f"{rbase}_{i}.json"), "recover")
with open(out, "w", encoding="utf-8") as f:
    json.dump(allr, f, indent=2, ensure_ascii=False)
c = Counter(r.get("label", r.get("cls_label")) for r in allr)
print(f"  -> {out}: {len(allr)} records | label histogram: {dict(c)}")
PY

echo "=== done -> $OUT  (logs: $dir/${base}_*.log , $dir/${rbase}_*.log) ==="
exit "$fail"
