import argparse
import ntpath
import os
import random
import re
import shutil
import time
from typing import List, Optional, Tuple

import torch
import transformers
from transformers import AutoTokenizer

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

from utils.file_utils import decode_response, read_txt_file, unique_filename_by_size
from utils.llava_utils import load_image

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp", ".tiff", ".tif"}
QUALITY_IN_DESC_PATTERN = re.compile(r"(?:^|,)\s*quality\s*-\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
QUALITY_GENERIC_PATTERN = re.compile(r"\bquality\b\s*[:=-]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
DESC_LINE_PATTERN = re.compile(r"(?:image[_ ]?description|description)\s*:\s*(.+)", re.IGNORECASE)


def get_filepaths(directory: str) -> List[str]:
    file_paths: List[str] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            low_path = filepath.lower()
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            if "_depth" in low_path or "_mask" in low_path or "(copy" in low_path:
                continue
            file_paths.append(filepath)
    return sorted(file_paths)


def split_paths_by_part(paths: List[str], which_part: int, num_parts: int) -> List[str]:
    if which_part == 0:
        return paths

    if num_parts <= 0:
        raise ValueError(f"num_parts must be > 0, got {num_parts}")
    if which_part < 1 or which_part > num_parts:
        raise ValueError(f"which_part must be in [1, {num_parts}] or 0, got {which_part}")

    total = len(paths)
    seg_len = total / num_parts
    start_idx = int((which_part - 1) * seg_len)
    end_idx = int(which_part * seg_len)
    if which_part == num_parts:
        end_idx = total

    return paths[start_idx:end_idx]


def load_llava(model_path: str, device_id: int):
    kwargs = {
        "device_map": device_id,
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


def get_llava_prompt(model, qs: str, conv_mode: str = "v1") -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def generate_batch_answers(
    model,
    tokenizer,
    image_processor,
    images,
    prompts: List[str],
    conv_mode: str = "v1",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
) -> List[str]:
    if len(images) == 0:
        return []

    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same length")

    image_sizes = [im.size for im in images]
    image_tensor = process_images(images, image_processor, model.config).to(
        model.device,
        dtype=torch.float16,
    )

    full_prompts = [get_llava_prompt(model, qs, conv_mode=conv_mode) for qs in prompts]

    tokenized = [
        tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for p in full_prompts
    ]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x.squeeze(0) for x in tokenized],
        batch_first=True,
        padding_value=pad_token_id,
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        attention_mask=(input_ids != pad_token_id).long(),
        pad_token_id=pad_token_id,
    )

    return [x.strip() for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


def extract_quality(output_text: str) -> Tuple[Optional[float], str]:
    description = ""

    try:
        answer_json, _ = decode_response(output_text)
        description = answer_json.get("Image description", "")
    except Exception:
        description = ""

    if not description:
        desc_match = DESC_LINE_PATTERN.search(output_text)
        if desc_match:
            description = desc_match.group(1).strip()

    match = QUALITY_IN_DESC_PATTERN.search(description)
    if not match:
        match = QUALITY_GENERIC_PATTERN.search(description)
    if not match:
        match = QUALITY_GENERIC_PATTERN.search(output_text)

    if not match:
        return None, description

    try:
        quality = float(match.group(1))
    except ValueError:
        return None, description

    return quality, description


def build_destination_path(
    src_path: str,
    input_dir: str,
    bad_dir: str,
    keep_structure: bool,
) -> str:
    if keep_structure:
        rel_path = os.path.relpath(src_path, input_dir)
        dst_path = os.path.join(bad_dir, rel_path)
    else:
        _, fname = ntpath.split(src_path)
        dst_path = os.path.join(bad_dir, fname)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    return unique_filename_by_size(dst_path, src_path)


def process_single_result(
    image_path: str,
    output: str,
    args: argparse.Namespace,
    counts: dict,
) -> None:
    quality, _ = extract_quality(output)
    if quality is None:
        counts["parse_failed"] += 1
        print(f"[WARN] quality parse failed: {image_path}")
        print(output)
        return

    if quality < args.threshold:
        dst_path = build_destination_path(
            src_path=image_path,
            input_dir=args.input_dir,
            bad_dir=args.bad_dir,
            keep_structure=(not args.flat_output),
        )
        if args.dry_run:
            print(f"[BAD] q={quality:.3f} {image_path} -> {dst_path} (dry-run)")
        else:
            shutil.move(image_path, dst_path)
            print(f"[BAD] q={quality:.3f} {image_path}")
        counts["moved"] += 1
    else:
        counts["kept"] += 1
        if args.verbose:
            print(f"[OK] q={quality:.3f} {image_path}")


def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}")

    device_id = int(args.device)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_id))
    device = torch.device(f"cuda:{device_id}")

    model, image_processor, tokenizer = load_llava(args.model_path, device_id)
    model = model.to(device)

    paths = get_filepaths(args.input_dir)
    # paths = split_paths_by_part(paths, args.which_part, args.num_parts)
    total = len(paths)

    if total == 0:
        print("[WARN] No image files found.")
        return

    prompts = read_txt_file(args.prompt_list)
    if not prompts:
        prompts = ["The image is a human face image. Is it real or fake? Why?"]

    counts = {
        "moved": 0,
        "kept": 0,
        "parse_failed": 0,
        "infer_failed": 0,
        "load_failed": 0,
    }

    print(
        f"[INFO] Start quality search: total={total}, threshold={args.threshold}, "
        f"batch_size={args.batch_size}, which_part={args.which_part}, num_parts={args.num_parts}"
    )

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_paths = paths[start:end]
        batch_t0 = time.time()

        loaded_images = []
        loaded_paths = []
        batch_prompts = []

        for path in batch_paths:
            try:
                loaded_images.append(load_image(path))
                loaded_paths.append(path)
                batch_prompts.append(random.choice(prompts))
            except Exception as e:
                counts["load_failed"] += 1
                print(f"[ERR] image load failed: {path} | {e}")

        if len(loaded_paths) == 0:
            if not args.verbose and end % args.log_every == 0:
                print(
                    f"[PROGRESS] {end}/{total} processed, moved={counts['moved']}, "
                    f"kept={counts['kept']}, parse_failed={counts['parse_failed']}, "
                    f"infer_failed={counts['infer_failed']}, load_failed={counts['load_failed']}"
                )
            batch_elapsed = time.time() - batch_t0
            print(f"[BATCH] {start + 1}-{end}/{total} time={batch_elapsed:.2f}s loaded=0")
            continue

        try:
            outputs = generate_batch_answers(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                images=loaded_images,
                prompts=batch_prompts,
                conv_mode="v1",
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )

            for path, output in zip(loaded_paths, outputs):
                process_single_result(path, output, args, counts)

        except Exception as e:
            print(f"[ERR] batch inference failed for {len(loaded_paths)} images: {e}")

        batch_elapsed = time.time() - batch_t0
        print(f"[BATCH] {start + 1}-{end}/{total} time={batch_elapsed:.2f}s loaded={len(loaded_paths)}")

        if not args.verbose and end % args.log_every == 0:
            print(
                f"[PROGRESS] {end}/{total} processed, moved={counts['moved']}, "
                f"kept={counts['kept']}, parse_failed={counts['parse_failed']}, "
                f"infer_failed={counts['infer_failed']}, load_failed={counts['load_failed']}"
            )

    print(
        f"[DONE] total={total}, moved={counts['moved']}, kept={counts['kept']}, "
        f"parse_failed={counts['parse_failed']}, infer_failed={counts['infer_failed']}, "
        f"load_failed={counts['load_failed']}"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Find low-quality images with LLaVA and move them to another folder."
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_0",
        help="Path to LLaVA model checkpoint.",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="/datasets/work/vLLM/data/lowp_real_4+13fmt/real",
        help="Input root directory containing images.",
    )
    p.add_argument(
        "--bad_dir",
        type=str,
        default="/datasets/work/vLLM/data/lowp_real_4+13fmt/real_bad",
        help="Destination root directory for low-quality images.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Move image if quality < threshold.",
    )
    p.add_argument(
        "--prompt_list",
        type=str,
        default="playground/prompts.txt",
        help="Path to newline-separated prompts.",
    )
    p.add_argument(
        "--which_part",
        type=int,
        default=0,
        help="0 = all data. Otherwise choose 1..num_parts.",
    )
    p.add_argument(
        "--num_parts",
        type=int,
        default=1,
        help="Number of parts used when which_part > 0.",
    )
    p.add_argument("--batch_size", type=int, default=80, help="Batch size for LLaVA inference.")
    p.add_argument("--device", type=int, default=0, help="CUDA device id.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--flat_output",
        action="store_true",
        help="Do not keep input subfolder structure under bad_dir.",
    )
    p.add_argument("--dry_run", action="store_true", help="Only print moves, do not move files.")
    p.add_argument("--verbose", action="store_true", help="Print per-image OK logs.")
    p.add_argument("--log_every", type=int, default=100, help="Progress log interval.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
