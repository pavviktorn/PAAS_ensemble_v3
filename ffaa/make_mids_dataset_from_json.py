import ntpath
import os
import re
import json
import time
import random
import argparse
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

from get_label import get_label_all


# ---------------------------
# Config / loading utilities
# ---------------------------

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


def get_llava_prompt(model, qs: str, conv_mode: str = "v1") -> str:
    """Build a single LLaVA prompt string for an image + question."""
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
    prompt = conv.get_prompt()
    return prompt

# ---------------------------
# Dataset & Collate
# ---------------------------

from typing import NamedTuple
@dataclass
class Sample:
    rec_id: str
    image_path: Path
    base_prompt: str
    cls_label: int

class ImageJSONDataset(Dataset):
    """
    Expects an input JSON containing a list of records:
      { "id": "...", "image": "relative/or/absolute/path.jpg", ... }
    """

    def __init__(
        self,
        input_json: str,
        image_root: str = "",
        prompt_list_path: str = "playground/prompts.txt",
        strict_exists: bool = False,
        which_part: int = 0,
    ):
        super(ImageJSONDataset, self).__init__()
        self.items: List[Sample] = []
        self.strict_exists = strict_exists

        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "Input JSON must be a list of records."

        if which_part != 0:
            data_len = len(data)
            seg_len = data_len / 7
            start_id = int((which_part-1)*seg_len)
            end_id = int(which_part*seg_len)
            if end_id > data_len - 1:
                end_id = data_len - 1
            data = data[start_id:end_id]
            data = random.sample(data, int(len(data)*0.5)) #zzzzzz

        image_root_path = Path(image_root) if image_root else Path(".")
        prompts = read_txt_file(prompt_list_path)
        if not prompts:
            prompts = ['The image is a human face image. Is it real or fake? Why?']

        for idx, item in enumerate(data, start=1):
            rec_id = item.get("id")
            rel_img = item.get("image")
            if rel_img is None:
                continue
            img_path = Path(rel_img)
            if not img_path.is_absolute():
                img_path = image_root_path / img_path
            if strict_exists and not img_path.exists():
                continue
            base_prompt = random.choice(prompts)

            conv = item.get("conversations")
            resp_json, _ = decode_response(conv[1]['value'])
            if resp_json['Analysis result'].lower() == 'real':
                cls_label = 0
            else:
                cls_label = 1

            # cls_label
            self.items.append(Sample(rec_id=rec_id, image_path=img_path,
                                     base_prompt=base_prompt, cls_label=cls_label))

        self._raw_records = data

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> Sample:
        s = self.items[i]
        return s

    @property
    def raw_records(self) -> List[Dict[str, Any]]:
        return self._raw_records

def collate_fn(samples: List[Sample]) -> Dict[str, Any]:
    rec_ids: List[str] = []
    paths: List[str] = []
    images: List[Any] = []
    base_prompts: List[str] = []
    cls_labels: List[int] = []
    for s in samples:
        rec_ids.append(s.rec_id)
        paths.append(str(s.image_path))
        images.append(load_image(str(s.image_path)))
        base_prompts.append(s.base_prompt)
        cls_labels.append(s.cls_label)
    return {"rec_ids": rec_ids, "paths": paths, "images": images,
            "base_prompts": base_prompts, "cls_labels": cls_labels}

# ---------------------------
# Batched generation helpers
# ---------------------------

def _tokenize_batch_with_images(model, tokenizer, prompts: List[str]) -> torch.Tensor:
    # Convert textual prompts (already containing image tokens) to padded input_ids
    toks = tokenizer_image_token(
        prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    )
    if isinstance(toks, torch.Tensor):
        input_ids = toks
    else:
        input_ids = toks["input_ids"]
    return input_ids.to(model.device)

def _format_prompts_with_conv(model, qs_list: List[str], conv_mode: str) -> List[str]:
    return [get_llava_prompt(model, qs, conv_mode) for qs in qs_list]

@torch.inference_mode()
def generate_batch_with_conditionals(
    model,
    tokenizer,
    image_processor,
    images: List[Any],
    prompts: List[str],
    conv_mode: str = "v1",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    per_sample_generate_num: int = 3,
) -> List[List[str]]:
    """
    Returns a list (length B) where each item is a list of 'per_sample_generate_num' outputs for that image.
    Pass 1: base prompts (provided)
    Pass 2 & 3: condition prompts chosen per-sample based on pass-1 JSON 'Analysis result'
    """
    assert len(images) == len(prompts), "images and prompts must align"
    B = len(images)

    # Preprocess all images together once
    image_sizes = [im.size for im in images]
    image_tensor = process_images(
        images, image_processor, model.config
    ).to(model.device, dtype=torch.float16)  # [B, C, H, W]

    # Helper: build a batch of input_ids from a list of question strings
    def build_input_ids(qs_list: List[str]) -> torch.Tensor:
        ids = []
        for qs in qs_list:
            full_prompt = get_llava_prompt(model, qs, conv_mode)  # inserts image tokens, conv template
            ids.append(
                tokenizer_image_token(
                    full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
            )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x.squeeze(0) for x in ids], batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(model.device)
        return input_ids

    # Prepare containers
    all_outputs = [[] for _ in range(B)]
    condition_prompt = "This is a _ human face. What evidence do you have?"

    # --------
    # Pass 1
    # --------
    input_ids = build_input_ids(prompts)
    out_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded_1 = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    for b in range(B):
        all_outputs[b].append(decoded_1[b].strip())

    # Decide per-sample branch for pass 2 & 3 based on pass-1 result
    # (mirrors your single-image adaptive logic)
    prompts_2: List[str] = []
    prompts_3: List[str] = []
    for b in range(B):
        # Use your JSON decoder to read 'Analysis result'
        resp_json, _ = decode_response(all_outputs[b][0])
        if len(resp_json) != 5:
            prompts_2.append(condition_prompt.replace("_", "fake"))
            prompts_3.append(condition_prompt.replace("_", "real"))
        elif resp_json["Analysis result"].lower() == "real":
            prompts_2.append(condition_prompt.replace("_", "fake"))
            prompts_3.append(condition_prompt.replace("_", "real"))
        else:
            prompts_2.append(condition_prompt.replace("_", "real"))
            prompts_3.append(condition_prompt.replace("_", "fake"))

    # --------
    # Pass 2
    # --------
    if per_sample_generate_num >= 2:
        input_ids = build_input_ids(prompts_2)
        out_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_2 = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        for b in range(B):
            all_outputs[b].append(decoded_2[b].strip())

    # --------
    # Pass 3
    # --------
    if per_sample_generate_num >= 3:
        input_ids = build_input_ids(prompts_3)
        out_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_3 = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        for b in range(B):
            all_outputs[b].append(decoded_3[b].strip())

    # Ensure each sample has exactly per_sample_generate_num strings
    for b in range(B):
        all_outputs[b] = all_outputs[b][:per_sample_generate_num]

    return all_outputs

# ---------------------------
# Orchestration
# ---------------------------

def main(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    device_id = int(args.device)
    device = resolve_cuda_device(device_id)

    # llava_groups = {
    #     "mistral": "checkpoints/effaa-llava-mistral-7b-lora_1",
    #     'phi': "checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_0",
    # }
    # model_path = llava_groups["mistral"]
    model_path = args.model_path

    model, image_processor, tokenizer = load_llava(model_path, device_id)
    model = model.to(device)

    dataset = ImageJSONDataset(
        input_json=args.input_json,
        image_root=args.image_root,
        prompt_list_path=args.prompt_list,
        strict_exists=args.strict_exists,
        which_part=args.which_part,
    )
    if len(dataset) == 0:
        print("[WARN] No valid records found. Exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    raw_records = dataset.raw_records
    data_real_easy_json: List[Dict[str, Any]] = []
    data_real_hard_json: List[Dict[str, Any]] = []
    data_pad_easy_json: List[Dict[str, Any]] = []
    data_pad_hard_json: List[Dict[str, Any]] = []
    data_df_easy_json: List[Dict[str, Any]] = []
    data_df_hard_json: List[Dict[str, Any]] = []
    data_makeup_easy_json: List[Dict[str, Any]] = []
    data_makeup_hard_json: List[Dict[str, Any]] = []

    total = len(dataset)
    seen = 0
    n_failed = 0

    for batch in loader:
        rec_ids: List[str] = batch["rec_ids"]
        paths: List[str] = batch["paths"]
        images: List[Any] = batch["images"]
        base_prompts: List[str] = batch["base_prompts"]
        cls_labels: List[int] = batch["cls_labels"]

        t0 = time.time()

        # try:
        with torch.no_grad():
            answers_3_per_image = generate_batch_with_conditionals(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                images=images,
                prompts=base_prompts,
                conv_mode="v1",
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                per_sample_generate_num=args.mistral_generations,
            )
        # except Exception as e:
        #     print(f"[ERROR] Failed a batch: {e}")
        #     answers_3_per_image = [["", "", ""] for _ in images]

        for rid, pth, cls_label, answers3 in zip(rec_ids, paths, cls_labels, answers_3_per_image):
            answers_result = []
            processed_answers = []
            cls_outputs = []
            all_ok = True
            for answer in answers3:
                answer_json, _ = decode_response(answer)
                if len(answer_json) != 5:
                    all_ok = False
                    break
                else:
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

            if ("sel_for_mids/fake/deepfake/DFGC-2022/DetectionDataset" in pth):
                print((f"Ambiguos label: {pth}"))
                continue

            label = get_label_all(pth)
            if all_ok == False:
                from shutil import copyfile
                head, fname = ntpath.split(pth)
                err_path = r'/datasets/work/vLLM/data/no_delete_mids_train/fmt_error_all'
                if label == 0:
                    dst_path = os.path.join(err_path, 'real')
                elif label == 1:
                    dst_path = os.path.join(err_path, 'fake', 'pad')
                elif label == 2:
                    dst_path = os.path.join(err_path, 'fake', 'deepfake')
                elif label == 3:
                    dst_path = os.path.join(err_path, 'fake', 'makeup')
                else:
                    continue

                if not os.path.exists(dst_path):
                    os.makedirs(dst_path, exist_ok=True)
                dst_path = os.path.join(dst_path, fname)
                dst_path = unique_filename_by_size(dst_path, pth)
                copyfile(pth, dst_path)

                n_failed += 1
                print((f"Error Format: {pth}"))
                continue

            out_item = {
                "id": rid,
                "image": str(Path(pth).resolve()),
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

            is_pad = (label == 1)
            is_makeup = (label == 3)
            if len(set(answers_result)) == 1:
                qs_difficulty = 'easy'
            else:
                qs_difficulty = 'hard'

            if cls_label == 0: #real
                if qs_difficulty == 'easy':
                    data_real_easy_json.append(out_item)
                else:
                    data_real_hard_json.append(out_item)
            elif cls_label == 1 and is_makeup is True:
                if qs_difficulty == 'easy':
                    data_makeup_easy_json.append(out_item)
                else:
                    data_makeup_hard_json.append(out_item)
            elif cls_label == 1 and is_pad is True:
                if qs_difficulty == 'easy':
                    data_pad_easy_json.append(out_item)
                else:
                    data_pad_hard_json.append(out_item)
            elif cls_label == 1 and is_pad is False and is_makeup is False:
                if qs_difficulty == 'easy':
                    data_df_easy_json.append(out_item)
                else:
                    data_df_hard_json.append(out_item)
            else:
                continue

        seen += len(rec_ids)
        print(f"[{args.which_part}/{seen}/{total}] {100.0*seen/total:.1f}% done, time:{time.time() - t0:.1f}s")
        # if seen > 0:
        #     break

    print(f"[failed:{n_failed}/{total}]")

    head, fname = ntpath.split(args.output_json)
    fn = fname.split('.')[0]

    out_path = os.path.join(head, f"{fn}_real_easy_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_real_easy_json, f, ensure_ascii=False, indent=2)
    print(f"[data_real_easy_json:{len(data_real_easy_json)}]")

    out_path = os.path.join(head, f"{fn}_real_hard_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_real_hard_json, f, ensure_ascii=False, indent=2)
    print(f"[data_real_hard_json:{len(data_real_hard_json)}]")

    out_path = os.path.join(head, f"{fn}_pad_easy_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_pad_easy_json, f, ensure_ascii=False, indent=2)
    print(f"[data_pad_easy_json:{len(data_pad_easy_json)}]")

    out_path = os.path.join(head, f"{fn}_pad_hard_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_pad_hard_json, f, ensure_ascii=False, indent=2)
    print(f"[data_pad_hard_json:{len(data_pad_hard_json)}]")

    out_path = os.path.join(head, f"{fn}_df_easy_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_df_easy_json, f, ensure_ascii=False, indent=2)
    print(f"[data_df_easy_json:{len(data_df_easy_json)}]")

    out_path = os.path.join(head, f"{fn}_df_hard_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_df_hard_json, f, ensure_ascii=False, indent=2)
    print(f"[data_df_hard_json:{len(data_df_hard_json)}]")

    out_path = os.path.join(head, f"{fn}_makeup_easy_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_makeup_easy_json, f, ensure_ascii=False, indent=2)
    print(f"[data_makeup_easy_json:{len(data_makeup_easy_json)}]")

    out_path = os.path.join(head, f"{fn}_makeup_hard_{args.which_part}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_makeup_hard_json, f, ensure_ascii=False, indent=2)
    print(f"[data_makeup_hard_json:{len(data_makeup_hard_json)}]")

    print(f"[INFO] Wrote {total} records to {out_path}")

# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch LLaVA inference with conditional prompts per image (GPU)."
    )
    p.add_argument("--model_path", type=str, default="checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_0",
                   help="Path to vLLM.")
    p.add_argument("--which_part", type=int, default=1,
                   help="Dataset shard index. Use 0 for all data, otherwise 1~7.")
    p.add_argument("--input_json", type=str, default="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext.json",
                   help="Path to the input dataset JSON (list of objects).")
    p.add_argument("--image_root", type=str, default="/datasets/newout",
                   help="Optional root to prefix each 'image' path from JSON.")
    p.add_argument("--output_json", type=str, default="/datasets/newout/vqa_info_2+13+4+3_fmt/temp/mids.json",
                   help="Where to write the augmented JSON with 'generated' field.")
    p.add_argument("--prompt_list", type=str, default="playground/prompts.txt",
                   help="Path to a newline-separated list of prompts.")
    p.add_argument("--strict_exists", action="store_true",
                   help="Skip records whose image files are missing.")

    # batching / performance
    p.add_argument("--batch_size", type=int, default=140, help="Images per batch.")
    p.add_argument("--num_workers", type=int, default=16, help="DataLoader workers for image I/O.")
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
