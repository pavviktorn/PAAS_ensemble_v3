#!/usr/bin/env bash
# Train A0-A3 with the 9-class (3 true x 3 claim) scheme on mids9.json (fixed-template claims,
# true label from FFAA get_label_all), then eval on the 30K/46-subset held-out set (full-image,
# fixed-answer; 9-class auto-detected) and compare to the existing 4-class A0-A3 results.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NINE=runs/real/9class
H=runs/real/heldout3
mkdir -p "$NINE/results9"
LOG="$NINE/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

# --- build 9-class training data ---
if [ ! -f runs/real/mids9/mids9.json ]; then
  say "=== building mids9.json (get_label_all -> 3-way true; 3 fixed-template claims) ==="
  python -u scripts/build_mids9.py >> "$NINE/build.log" 2>&1
  say "build exit=$? ; $(grep -E 'built|true-type' "$NINE/build.log" | tail -2 | tr '\n' ' ')"
fi
TR=runs/real/mids9/mids9.json; VA=runs/real/mids9/mids9_val.json
COMMON="train_data_path=$TR val_data_path=$VA num_classes=9 samples_per_image=3 skip_missing_images=true epochs=1 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=8 save_every_steps=2000"

# --- train A0-A3 (9-class), one per GPU ---
mapfile -t G < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -4 | cut -d, -f1 | tr -d ' ')
say "GPUs for training: ${G[*]}"
say "=== training A0-A3 (9-class) ==="
CUDA_VISIBLE_DEVICES=${G[0]} python -u -m mids_plus.train --config configs/mids_original.yaml \
  --set $COMMON output_dir=$NINE/A0_baseline > "$NINE/train_A0.log" 2>&1 & P0=$!
CUDA_VISIBLE_DEVICES=${G[1]:-${G[0]}} python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false alignment_weight=0 uniformity_weight=0 query_orth_weight=0 tune_layer_norm=false \
  output_dir=$NINE/A1_svd > "$NINE/train_A1.log" 2>&1 & P1=$!
CUDA_VISIBLE_DEVICES=${G[2]:-${G[0]}} python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false query_orth_weight=0 \
  output_dir=$NINE/A2_svdgend > "$NINE/train_A2.log" 2>&1 & P2=$!
CUDA_VISIBLE_DEVICES=${G[3]:-${G[0]}} python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON output_dir=$NINE/A3_midspp > "$NINE/train_A3.log" 2>&1 & P3=$!
wait $P0; say "A0 train exit=$?"
wait $P1; say "A1 train exit=$?"
wait $P2; say "A2 train exit=$?"
wait $P3; say "A3 train exit=$?"

# --- eval on the 46 held-out subsets (full-image, fixed-answer; 9-class auto) ---
SPECS=""
for f in "$H"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done
GP=${G[0]}
for tag in A0_baseline A1_svd A2_svdgend A3_midspp; do
  if [ -f "$NINE/$tag/best.pt" -o -f "$NINE/$tag/last.pt" ]; then
    say "eval $tag (9-class)"
    CUDA_VISIBLE_DEVICES=$GP python -u scripts/eval_fixed_answers.py --model-type mids_plus \
      --checkpoint "$(ckpt $NINE/$tag)" --full-image --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "${tag}_9c" --out "$NINE/results9/${tag}.json" >> "$NINE/eval.log" 2>&1
    say "  eval $tag exit=$?"
  fi
done

# --- comparison report: 9-class vs 4-class (same held-out buckets) ---
say "=== building 9-class vs 4-class comparison ==="
python -u - <<'PY' > "$NINE/NINE_VS_FOUR_REPORT.md" 2>>"$NINE/build.err"
import json, os, math
H="runs/real/heldout3"; NINE="runs/real/9class/results9"
man=json.load(open(f"{H}/manifest.json"))
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),("A3 MIDS++","A3_midspp")]
def loadj(p): return json.load(open(p))["per_set"] if os.path.exists(p) else {}
def merged4(tag):  # 4-class: filtered real (results_realf) + pad/deepfake (results)
    real=loadj(f"{H}/results_realf/{tag}.json"); full=loadj(f"{H}/results/{tag}.json")
    m=dict(real)
    for k,v in full.items():
        if man.get(k,{}).get("group") in ("PAD","DEEPFAKE"): m[k]=v
    return m
def nine(tag): return loadj(f"{NINE}/{tag}.json")
def n(ps,k): return ps[k]["n"] if k in ps else 0
def grec(ps,g):
    ks=[s for s in man if man[s]["group"]==g]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else float('nan')
def frec(ps):
    ks=[s for s in man if man[s]["group"] in ("PAD","DEEPFAKE")]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else float('nan')
print("# 9-class vs 4-class on the same 30K full-image held-out set (filtered real)\n")
print("9-class: 3 true x 3 claim, fixed-template claims, true label via get_label_all (makeup->pad). "
      "4-class: existing models (MLLM-answer-trained). Both eval full-image, MLLM-free, same buckets. "
      "Recall per group (PAD = anti-spoof recall, the priority).\n")
print("| Model | scheme | PAD-rec | DEEPFAKE-rec | FAKE-rec | REAL-rec | Balanced ACC |")
print("|---|---|---|---|---|---|---|")
for nm,tag in MODELS:
    for scheme,ps in [("4-class",merged4(tag)),("9-class",nine(tag))]:
        if not ps: print(f"| {nm} | {scheme} | (missing) |"); continue
        pr,dr,fr,rr=grec(ps,"PAD"),grec(ps,"DEEPFAKE"),frec(ps),grec(ps,"REAL")
        print(f"| {nm} | {scheme} | {pr:.2f} | {dr:.2f} | {fr:.2f} | {rr:.2f} | {(rr+fr)/2:.2f} |")
print("\n## PAD per-attack-type recall (9-class)\n")
padsubs=sorted([s for s in man if man[s]["group"]=="PAD"], key=lambda s:-n(nine("A3_midspp"),s))
print("| attack | n | " + " | ".join(nm for nm,_ in MODELS) + " |")
print("|"+" --- |"*(len(MODELS)+2))
for s in padsubs:
    print(f"| {s} | {n(nine('A3_midspp'),s)} | " + " | ".join(f"{nine(t).get(s,{}).get('acc',float('nan'))*100:.2f}" if s in nine(t) else "-" for _,t in MODELS) + " |")
PY
cat "$NINE/NINE_VS_FOUR_REPORT.md" >> "$LOG"
say "=== 9CLASS DONE ==="
touch "$NINE/DONE_9CLASS"
