import ntpath
import os
import re
import json
import time
import random
import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.utils.data as data
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, BitsAndBytesConfig

# --- LLaVA + your helpers
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import process_images, tokenizer_image_token
from utils.file_utils import *
from utils.llava_utils import *
from utils.cuda_utils import resolve_cuda_device

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.distributed as dist


# ---------------------------
# Config / loading utilities
# ---------------------------

_INTERRUPTED = {"flag": False}


def _install_sigint_handler():
    """First Ctrl+C finishes the current image and saves; a second one force-quits."""
    import signal

    def _handler(signum, frame):
        _INTERRUPTED["flag"] = True
        print("\n[INFO] Interrupt received — finishing current image, then saving. "
              "Press Ctrl+C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _handler)


def write_json(obj, path):
    """Atomically write obj as JSON to path (tmp file + os.replace).

    Creates parent dirs as needed. On failure, falls back to writing in the
    current directory under the same basename. Returns the path actually written.
    """
    head = os.path.dirname(path)
    try:
        if head:
            os.makedirs(head, exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return path
    except Exception as e:
        fallback_path = os.path.basename(path)
        print(f"[ERROR] Failed to write {path}: {e}. Falling back to {fallback_path}")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return fallback_path


def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return sorted(file_paths)

def load_llava(model_path: str, device_id: int):
    device_map = device_id
    kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
        "use_flash_attention_2": True,
    }
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign=True)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model, image_processor, tokenizer

def get_llava_prompt(model, qs, conv_mode):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def get_llava_answer(model, tokenizer, image_processor, image, prompt,
                     temperature, top_p, num_beams, max_new_tokens,
                     per_sample_generate_num, conv_mode='v1'):
    image_size = [image.size]
    image_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    outputs = []

    # prepare prompts
    condition_prompt = 'This is a _ human face. What evidence do you have?'
    prompts = []
    prompts.append(prompt)

    with torch.inference_mode():
        for i in range(per_sample_generate_num):
            qs = prompts[i]
            prompt = get_llava_prompt(model, qs, conv_mode)
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .to(model.device)
            )
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_size,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
                pad_token_id=tokenizer.pad_token_id,
            )

            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if i == 0:
                response_json, _ = decode_response(output)
                if response_json['Analysis result'].lower() == 'real':
                    prompts.append(condition_prompt.replace('_', 'fake'))
                    prompts.append(condition_prompt.replace('_', 'real'))
                else:
                    prompts.append(condition_prompt.replace('_', 'real'))
                    prompts.append(condition_prompt.replace('_', 'fake'))
            outputs.append(output)

        return outputs


def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    _install_sigint_handler()

    device_id = int(args.device)
    device = resolve_cuda_device(device_id)

    model_path = args.model_path

    model, image_processor, tokenizer = load_llava(model_path, device_id)
    model = model.to(device)

    data_json: List[Dict[str, Any]] = []

    out_head, out_fname = ntpath.split(args.output_json)
    out_fn = out_fname.split('.')[0]
    out_path = os.path.join(out_head, f"{out_fn}_{args.which_part}.json")

    # --- collect images already processed in prior runs (to skip) ---
    # args.resume is a comma-separated list of output JSON filenames living
    # alongside --output_json (or absolute paths). Their 'image' fields are
    # the resolved paths to skip.
    processed_paths = set()
    if args.resume:
        for name in args.resume.split(","):
            name = name.strip()
            if not name:
                continue
            json_path = name if os.path.isabs(name) else os.path.join(out_head, name)
            if not os.path.exists(json_path):
                print(f"[RESUME] Processed file not found, skipping: {json_path}")
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                before = len(processed_paths)
                for item in existing:
                    img = item.get("image")
                    if img:
                        processed_paths.add(img)
                print(f"[RESUME] {json_path}: +{len(processed_paths) - before} images "
                      f"({len(existing)} records)")
            except Exception as e:
                print(f"[WARN] Could not load {json_path}: {e}")
        print(f"[RESUME] {len(processed_paths)} processed images will be skipped.")

    file_paths = get_filepaths(args.input_dir)

    full_file_paths = []
    n_skipped = 0
    for img_path in file_paths:
        head, fname = ntpath.split(img_path)
        if os.path.isdir(img_path) is False and (fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png')):
            if "sel_for_mids/fake/deepfake/DFGC-2022/DetectionDataset" in img_path \
                or ("/fake" in img_path and "/real" in img_path) \
                or "PAD/dataset/mywebcam/real" in img_path:
                continue # bad label
            elif processed_paths and str(Path(img_path).resolve()) in processed_paths:
                n_skipped += 1
                continue  # already processed in a prior run
            else:
                full_file_paths.append(img_path)
    if n_skipped:
        print(f"[RESUME] Skipped {n_skipped} already-processed images while building the work list.")
    
    if args.which_part != -1:
        data_len = len(full_file_paths)
        seg_len = data_len / args.n_divided
        start_id = int(args.which_part * seg_len)
        end_id = int((args.which_part+1) * seg_len)
        if end_id > data_len:
            end_id = data_len - 1
        full_file_paths = full_file_paths[start_id:end_id]

    prompts = read_txt_file(args.prompt_list)
    if not prompts:
        prompts = ['The image is a human face image. Is it real or fake? Why?']

    total = len(full_file_paths)
    seen = 0
    n_failed = 0

    n_repeat = 10
    rid = 20000000
    for img_path in full_file_paths:
        if _INTERRUPTED["flag"]:
            print("[INFO] Stopping early due to interrupt; writing what we have.")
            break
        head, fname = ntpath.split(img_path)
        if os.path.isdir(img_path) is False and (fname.endswith('.jpg') or fname.endswith('.jpeg') or fname.endswith('.png')):
            rid += 1
            try:
                img_path = img_path.replace("\\", '/')
                base_prompt = random.choice(prompts)
                if '/real' in img_path:
                    cls_label = 0
                elif '/fake' in img_path:
                    cls_label = 1
                else:
                    continue

                resolved_img = str(Path(img_path).resolve())

                try:
                    image = load_image(img_path)
                except Exception as e:
                    n_failed += 1
                    print(f"[ERROR] Failed to load image: {img_path}, error: {e}")
                    continue

                t0 = time.time()
                answers_result = []
                processed_answers = []
                cls_outputs = []
                all_ok = True

                for repeat in range(n_repeat):
                    with torch.no_grad():
                        answers3 = get_llava_answer(
                            model=model,
                            tokenizer=tokenizer,
                            image_processor=image_processor,
                            image=image,
                            prompt=base_prompt,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            per_sample_generate_num=args.mistral_generations,
                            conv_mode="v1",
                        )

                    answers_result = []
                    processed_answers = []
                    cls_outputs = []
                    all_ok = True
                    for answer in answers3:
                        answer_json, _ = decode_response(answer)
                        # if len(answer_json) != 5:
                        #     all_ok = False
                        #     break
                        if 'Analysis result' in answer_json and 'Image description' in answer_json and 'Forgery reasoning' in answer_json:
                            answer_result = answer_json['Analysis result'].lower()
                            new_answer_json = {
                                'Image description': answer_json['Image description'],
                                'Forgery reasoning': answer_json['Forgery reasoning']
                            }
                            new_answer = "\n".join(f"{key}: {value}" for key, value in new_answer_json.items())
                            answers_result.append(answer_result)
                            processed_answers.append(new_answer)
                            if answer_result == 'real':
                                cls_out = 0
                            else:
                                cls_out = 1
                            cls_outputs.append(cls_out)
                        else:
                            all_ok = False
                            break

                    if all_ok == True:
                        break

                if all_ok == False:
                    try:
                        from shutil import copyfile
                        from get_label import get_label_all
                        head, fname = ntpath.split(img_path)
                        err_path = args.err_path
                        label = get_label_all(img_path)
                        dst_path = None
                        if label == 0:
                            dst_path = os.path.join(err_path, 'real')
                        elif label == 1:
                            dst_path = os.path.join(err_path, 'fake', 'pad')
                        elif label == 2:
                            dst_path = os.path.join(err_path, 'fake', 'deepfake')
                        elif label == 3:
                            dst_path = os.path.join(err_path, 'fake', 'makeup')

                        if dst_path is not None:
                            if not os.path.exists(dst_path):
                                os.makedirs(dst_path, exist_ok=True)
                            dst_path = os.path.join(dst_path, fname)
                            dst_path = unique_filename_by_size(dst_path, img_path)
                            copyfile(img_path, dst_path)
                    except Exception as e:
                        print(f"[WARN] Failed to archive bad-format image {img_path}: {e}")

                    n_failed += 1
                    print(f"Error Format: {img_path}")
                    continue

                if cls_label == 0:
                    try:
                        answer_json, _ = decode_response(answers3[0])
                        if 'Image description' not in answer_json:
                            print(f"Missing Image description: {img_path}")
                            continue

                        desc = answer_json.get('Image description', "")
                        data = {}

                        for part in desc.split(","):
                            part = part.strip()
                            if "-" in part:                      # ✅ ensure valid key-value format
                                key, value = part.split("-", 1)
                                data[key.strip()] = value.strip()

                        quality_str = data.get("quality")
                        # quality_value = float(quality_str) if quality_str else 0.0
                        # if quality_value < 0.5:
                        quality_value = quality_str.lower()
                        if quality_value in ["low", "poor"]:
                            print(f"Low quality real image: {img_path}, quality: {quality_value}")
                            continue
                    except Exception as e:
                        print(f"Error processing real image description: {img_path}, error: {e}")
                        continue

                if len(processed_answers) < 3 or len(answers_result) < 3 or len(cls_outputs) < 3:
                    n_failed += 1
                    print(f"[WARN] Incomplete answers for {img_path} (got {len(processed_answers)}), skipping.")
                    continue

                out_item = {
                    "id": rid,
                    "image": resolved_img,
                    "cls_label": cls_label,
                    "answers": [
                        {
                            "content": processed_answers[0],
                            "result": answers_result[0],
                            "label": 2*cls_label+cls_outputs[0]
                        },
                        {
                            "content": processed_answers[1],
                            "result": answers_result[1],
                            "label": 2*cls_label+cls_outputs[1]
                        },
                        {
                            "content": processed_answers[2],
                            "result": answers_result[2],
                            "label": 2*cls_label+cls_outputs[2]
                        },
                    ]
                }
                data_json.append(out_item)
                seen += 1
                print(f"[{args.which_part}/{seen}/{total}] {100.0*seen/total:.1f}% done, time:{time.time() - t0:.1f}s")

                if args.save_every > 0 and seen % args.save_every == 0:
                    try:
                        saved_to = write_json(data_json, out_path)
                        print(f"[CKPT] {seen}/{total} — saved {len(data_json)} records to {saved_to}")
                    except Exception as e:
                        print(f"[WARN] Checkpoint save failed: {e}")
            except Exception as e:
                n_failed += 1
                print(f"[ERROR] Unexpected failure on {img_path}: {e}")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    print(f"[failed:{n_failed}/{total}] [skipped(resumed):{n_skipped}]")

    # repeat = 20
    data_json_ex: List[Dict[str, Any]] = []
    for rep in range(args.repeat):
        data_json_ex += data_json

    out_path = write_json(data_json_ex, out_path)
    print(f"[data_json_ex:{len(data_json_ex)}]")

    print(f"[INFO] Wrote {len(data_json)} records ({len(data_json_ex)} with repeat={args.repeat}) to {out_path}")

# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch LLaVA inference with conditional prompts per image (GPU)."
    )
    p.add_argument("--model_path", type=str, default="checkpoints/effaa-llava-mistral-7b-lora_3",
                   help="Path to vLLM.")
    p.add_argument("--which_part", type=int, default=-1,
                   help="Dataset shard index. Use -1 for all data, otherwise 0~n_divided.")
    p.add_argument("--n_divided", type=int, default=8,
                   help="Dataset divided num. Use -1 for all data, otherwise 0~7.")
    p.add_argument("--input_dir", type=str, default="/datasets/work/vLLM/data/fmt_error_all",
                   help="Path to the input dataset JSON (list of objects).")
    p.add_argument("--image_root", type=str, default="/datasets/newout",
                   help="Optional root to prefix each 'image' path from JSON.")
    p.add_argument("--output_json", type=str, default="/datasets/newout/vqa_info_2+13+4+3_fmt/temp/mids_dir_err1.json",
                   help="Where to write the augmented JSON with 'generated' field.")
    p.add_argument("--prompt_list", type=str, default="playground/prompts.txt",
                   help="Path to a newline-separated list of prompts.")
    p.add_argument("--err_path", type=str, default="/datasets/work/vLLM/data/fmt_error2",
                   help="Dir to archive images this stage STILL fails to parse, organized by label. "
                        "Default kept for compatibility.")
    p.add_argument("--strict_exists", action="store_true",
                   help="Skip records whose image files are missing.")

    # batching / performance
    p.add_argument("--batch_size", type=int, default=140, help="Images per batch.")
    p.add_argument("--repeat", type=int, default=1, help="repeat num.")
    p.add_argument("--save_every", type=int, default=500,
                   help="Checkpoint accumulated results every N processed images (0 disables).")
    p.add_argument("--resume", type=str, default="",
                   help="Comma-separated list of already-processed output JSON filenames located "
                        "alongside --output_json (or absolute paths); images present in them are "
                        "skipped. e.g. 'mids_dir_err1_0.json,mids_dir_err1_2.json,mids_dir_err1_7.json'")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers for image I/O.")
    p.add_argument("--device", type=int, default=0, help="CUDA device id (e.g., 0, 1, ...).")

    # generation knobs
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--mistral_generations", type=int, default=3)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
# --which_part 1
