#!/usr/bin/env bash
# PAAS_ensemble_v3 -- end-to-end FFAA (MLLM+MIDS) fine-tuning, chaining all three steps.
# Faithful port of the user's __run_finetuning.sh, repointed at PAAS bundled model assets:
#   Step 1  LoRA-finetune LLaVA-Mistral-7B  -> merge_lora_weights  (-> merged MLLM)
#   Step 2  generate MIDS training data with the merged MLLM       (-> mids.json)
#   Step 3  train the MIDS 4-class head (warm-start)               (-> mids/*.pth)
#
# Runs on the GLOBAL python3.12 / transformers==4.37.2. Bundled CLIP/T5 + warm-start mids.pth are
# used; the pre-LoRA BASE MLLM and the training IMAGE DATA stay at their original locations (too
# large to bundle) -- override any path via the env vars below.
#
# Each step is gated so you can run any subset:
#   RUN_STEP1=1 RUN_STEP2=1 RUN_STEP3=1 bash train/run_finetuning.sh     # full chain (default)
#   RUN_STEP1=0 RUN_STEP2=0 RUN_STEP3=1 bash train/run_finetuning.sh     # just retrain MIDS head
set -e

############################################
# GLOBAL SETTINGS  (override via env)
############################################
PROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # project root = this script's directory
cd "$PROOT/ffaa"
export PYTHONPATH="$PROOT/ffaa:$PYTHONPATH"

DEVICE_MAP="${DEVICE_MAP:-localhost:0,1,2,3}"
RUN_STEP1="${RUN_STEP1:-0}"; RUN_STEP2="${RUN_STEP2:-0}"; RUN_STEP3="${RUN_STEP3:-1}"

# --- training log: tee everything (this script + the deepspeed/python subprocesses) to a file ---
LOG_DIR="${LOG_DIR:-$PROOT/runs/finetune_logs}"
mkdir -p "$LOG_DIR"
LOG="${LOG:-$LOG_DIR/run_finetuning_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "[run_finetuning] $(date '+%F %T') | logging to $LOG"

# CPU thread cap. deepspeed spawns one process per GPU and each defaults its OMP/MKL pool to the
# full core count -> 4 x 96 = 384 threads on 96 cores = every core pinned at 100%. Cap per-process
# threads instead: total busy cores ~= OMP_NUM_THREADS x num_gpus. Default 12 x 4 = ~48 cores (~50%
# of 96). Set OMP_NUM_THREADS=96 for full speed, =6 for ~25%.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# --- models (bundled where possible) ---
CLIP="$PROOT/base_models/clip-vit-large-patch14-336"
T5="$PROOT/base_models/t5-base"
Base_vLLM_path="${BASE:-/datasets/work/vLLM/FFAA-master/checkpoints/ffaa-mistral-7b}"
# Base_vLLM_path="${BASE:-$PROOT/weights/ffaa_llava_mids}"   # pre-LoRA base: place here or override BASE
Finetuned_vLLM_path="${LORA_OUT:-$PROOT/runs/mllm_lora}"
Merged_vLLM_path="${MERGED:-$PROOT/runs/mllm_merged}"
init_mids="${INIT:-$PROOT/runs/mids_head/mids_v1_20260711_121152/best.pth}"   # warm-start MIDS head (bundled)

# --- data (the user's generated VQA / image sources; external, override as needed) ---
all_json="${ALL_JSON:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json}"
base_json="${BASE_JSON:-$all_json}"                           # input for the make_mids_dataset_from_json variant
eval_json="${EVAL_JSON:-/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json}"
mids_json="${MIDS_JSON:-/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json}"
mids_eval_json="${MIDS_EVAL_JSON:-/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json}"

# Step-2 image sources / outputs (the user's original dirs; override as needed)
fb_input_dir="${FB_INPUT_DIR:-/datasets/work/vLLM/data/no_delete_mids_train}"
fb_output_json="${FB_OUTPUT_JSON:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir.json}"
image_dir1="${IMAGE_DIR1:-/datasets/work/vLLM/data/fmt_error_all}"
output_json1="${OUTPUT_JSON1:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err1.json}"
image_dir2="${IMAGE_DIR2:-/datasets/work/vLLM/data/fmt_error11}"
output_json2="${OUTPUT_JSON2:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err2.json}"
image_dir3="${IMAGE_DIR3:-/datasets/work/vLLM/data/no_delete_mids_train/miss_axonlabs_data_1_mids++_fastermodel}"
output_json3="${OUTPUT_JSON3:-/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix/mids_dir_err3.json}"

mids_out="${MIDS_OUT:-$PROOT/runs/mids_head}"

# multi-GPU sharding helper (one part per visible device), as in the original script
run_parts () {
  local cmd="$1"; local num_parts="$2"
  local raw_devices="${DEVICE_MAP#*:}"; local devices=(); local pids=(); local failed=0
  IFS=',' read -ra devices <<< "$raw_devices"; local n="${#devices[@]}"
  [ "$n" -eq 0 ] && { echo "No CUDA devices in DEVICE_MAP=$DEVICE_MAP" >&2; exit 1; }
  for i in $(seq 0 $((num_parts - 1))); do
    local gpu="${devices[$((i % n))]}"
    echo "  part $i/$((num_parts - 1)) on GPU $gpu"
    ( export CUDA_VISIBLE_DEVICES="$gpu"; eval "$cmd --which_part $i --device 0" ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

############################################
# STEP 1 — Finetune Mistral LoRA + merge
############################################
if [ "$RUN_STEP1" = "1" ]; then
  echo "==== Step 1: LoRA finetune LLaVA-Mistral + merge ===="
  deepspeed --master_port "${PORT1:-25642}" --include "$DEVICE_MAP" \
      llava/train/train_mem.py \
      --lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 1e-6 \
      --deepspeed ./scripts/zero3.json \
      --model_name_or_path "$Base_vLLM_path" --version v1 \
      --data_path "$all_json" --eval_data_path "$eval_json" \
      --image_folder /datasets/newout \
      --vision_tower "$CLIP" \
      --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
      --mm_use_im_start_end False --mm_use_im_patch_token False \
      --image_aspect_ratio pad --group_by_modality_length True --bf16 True \
      --output_dir "$Finetuned_vLLM_path" \
      --num_train_epochs "${EPOCHS1:-3}" \
      --per_device_train_batch_size "${BS1:-64}" --per_device_eval_batch_size 24 \
      --gradient_accumulation_steps 1 \
      --save_strategy steps --save_steps 500 --save_total_limit 3 \
      --learning_rate "${LR1:-1e-5}" --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
      --logging_steps 1 --tf32 True --model_max_length 2048 --gradient_checkpointing True \
      --dataloader_num_workers 32 --lazy_preprocess True --report_to none \
      --evaluation_strategy steps --eval_steps 500 --metric_for_best_model eval_loss --greater_is_better False
  python3.12 merge_lora_weights.py \
      --model-path "$Finetuned_vLLM_path" --model-base "$Base_vLLM_path" --save-model-path "$Merged_vLLM_path"
else
  echo "---- Step 1 skipped (RUN_STEP1=0); using merged MLLM at $Merged_vLLM_path ----"
fi

############################################
# STEP 2 — Build MIDS dataset with the merged MLLM
############################################
if [ "$RUN_STEP2" = "1" ]; then
  echo "==== Step 2: generate MIDS dataset (merged MLLM) ===="

  # --- variant A: from a VQA json ---
  # run_parts "python3.12 make_mids_dataset_from_json.py \
  #    --model_path $Merged_vLLM_path \
  #    --input_json $base_json --batch_size 140" 7

  # --- variant B: whole folder, batched ---
  run_parts "python3.12 make_mids_dataset_from_folder_batch.py \
      --model_path $Merged_vLLM_path --batch_size 140 \
      --input_dir $fb_input_dir \
      --output_json $fb_output_json \
      --n_divided 4" 4

  # --- variant C: folder one-by-one (dir1) ---
  run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
    --model_path $Merged_vLLM_path \
    --input_dir $image_dir1 --output_json $output_json1 \
    --n_divided 20 --repeat 1" 20

  # --- variant D: folder one-by-one (dir2) ---
  run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
    --model_path $Merged_vLLM_path \
    --input_dir $image_dir2 --output_json $output_json2 \
    --n_divided 20 --repeat 10" 20

  # # --- variant E: folder one-by-one (dir3) [ACTIVE] ---
  # run_parts "python3.12 make_mids_dataset_from_folder_onebyone.py \
  #     --model_path $Merged_vLLM_path \
  #     --input_dir $image_dir3 \
  #     --output_json $output_json3 \
  #     --n_divided 20 --repeat 10" 20

  python3.12 merge_mids_json.py            # merges the sharded parts -> $mids_json (as configured in that script)
  echo "  built MIDS dataset -> $mids_json"
else
  echo "---- Step 2 skipped (RUN_STEP2=0); using MIDS data at $mids_json ----"
fi

############################################
# STEP 3 — Train MIDS v1 (warm-start)
############################################
if [ "$RUN_STEP3" = "1" ]; then
  echo "==== Step 3: train MIDS head (warm-start from $init_mids) ===="
  deepspeed --master_port "${PORT3:-25635}" --include "$DEVICE_MAP" \
      train_mids_new.py \
      --hidden_dim 768 --version v1 \
      --image_model_path "$CLIP" --text_model_path "$T5" \
      --init_model_path "$init_mids" \
      --data_path "$mids_json" --val_data_path "$mids_eval_json" \
      --output_dir "$mids_out" \
      --per_device_train_batch_size "${BS3:-24}" --per_device_val_batch_size 8 \
      --learning_rate "${LR3:-1e-5}" --unfreeze_vision_encoder_last_layers 2 \
      --num_train_epochs "${EPOCHS3:-30}" --warmup_ratio 0.03 --weight_decay 1e-5 \
      --eval_every_steps "${EVAL_STEPS3:-5000}" --select_metric "${SELECT_METRIC3:-acc}"
  echo "  trained MIDS head -> $mids_out/*.pth"
fi

echo "==== run_finetuning.sh DONE ===="
echo "Deploy the merged MLLM ($Merged_vLLM_path) + chosen MIDS epoch into weights/ffaa_llava_mids/,"
echo "or point config FFAACfg.llava_dir / mids_path at them."
