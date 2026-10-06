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
from shutil import copyfile
from get_label import get_label_all
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn.functional as F
import transformers
transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPProcessor

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
from mids.selector import make_decision

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.distributed as dist


# ---------------------------
# Config / loading utilities
# ---------------------------

def get_filepaths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if "_depth" in filepath or "(copy" in filepath or "(_mask)" in filepath:
                continue
            if (
                    filename.endswith('.jpg') or filename.endswith('.jpeg')
                    or filename.endswith('.png') or filename.endswith('.bmp')
                    or filename.endswith('.jfif')
            ):
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
# MIDS loading & helpers
# ---------------------------

def load_mids(mids_path: str, device_id: int):
    from mids.mids_arch import MIDS
    model = MIDS()

    model_state_dict = model.state_dict()
    finetuned_state_dict = torch.load(mids_path)
    finetuned_state_dict = {
        key.replace("module.", ""): value for key, value in finetuned_state_dict.items()
    }
    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    return model.to(dtype=torch.float32, device=torch.device(f"cuda:{device_id}"))


def run_mids_for_one(
    mids_model,
    clip_processor: CLIPProcessor,
    t5_tokenizer: AutoTokenizer,
    image,
    answers: List[str],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """
    Mirror the logic from inference.py:
      - mask_result on each answer
      - compute difficulty (easy/hard)
      - choose N, M based on #answers
      - run MIDS and make_decision to get best answer index + scores
    """
    if len(answers) == 0:
        return None

    # mask answers
    answers_result: List[str] = []
    processed_answers: List[str] = []
    for ans in answers:
        masked_ans, ans_res = mask_result(ans)
        processed_answers.append(masked_ans)
        answers_result.append(ans_res)

    # easy or hard
    if len(set(answers_result)) == 1:
        qs_difficulty = "easy"
    else:
        qs_difficulty = "hard"

    # N, M follow inference.py
    if len(answers) == 3:
        N, M = 1, 1
    elif len(answers) == 2:
        N, M = 0, 1
    else:
        N, M = 0, 0

    with torch.inference_mode():
        input_image = clip_processor(images=image, return_tensors="pt")["pixel_values"]
        input_image = input_image.to(device)

        answer_ids = t5_tokenizer(
            processed_answers,
            return_tensors="pt",
            padding="longest",
            max_length=t5_tokenizer.model_max_length,
            truncation=True,
        )

        logits = mids_model(
            answer_ids.to(device),
            input_image,
            None,
            1,
            N,
            M,
        )["logits"]

        scores = F.softmax(logits, dim=2).squeeze(0)
        best_answer_idx, pred, match_score, forgery_score = make_decision(
            answers_result, scores
        )

    return {
        "best_answer_idx": int(best_answer_idx),
        "difficulty": qs_difficulty,
        "match_score": float(match_score),
        "forgery_score": float(forgery_score),
        "answers_result": answers_result,
    }


def build_mids_out_item(
    rec_id: str,
    image_path: str,
    cls_label: int,
    processed_answers: List[str],
    answers_result: List[str],
    cls_outputs: List[int],
) -> Dict[str, Any]:
    return {
        "id": rec_id,
        "image": str(Path(image_path).resolve()),
        "cls_label": cls_label,
        "answers": [
            {
                "content": processed_answers[0],
                "result": answers_result[0],
                "label": 2 * cls_label + cls_outputs[0],
            },
            {
                "content": processed_answers[1],
                "result": answers_result[1],
                "label": 2 * cls_label + cls_outputs[1],
            },
            {
                "content": processed_answers[2],
                "result": answers_result[2],
                "label": 2 * cls_label + cls_outputs[2],
            },
        ],
    }


def write_mids_out_item_json(image_path: str, out_item: Dict[str, Any]) -> None:
    json_path = str(Path(image_path).with_suffix(".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_item, f, ensure_ascii=False, indent=2)


# ---------------------------
# Dataset & Collate
# ---------------------------

input_paths = [
    # "/datasets/newout/PAD/dataset/3DMAD",
    # "/datasets/newout/PAD/dataset/CeFA",
    # "/datasets/newout/PAD/dataset/CelebA-Spoof",
    # "/datasets/newout/PAD/dataset/CSMAD",
    # "/datasets/newout/PAD/dataset/CVPR2023-ASF",
    # "/datasets/newout/PAD/dataset/datatang_sample",
    # "/datasets/newout/PAD/dataset/EPRA",
    # "/datasets/newout/PAD/dataset/from_nizar",
    # "/datasets/newout/PAD/dataset/HiFiMask",
    # "/datasets/newout/PAD/dataset/mydata",
    # "/datasets/newout/PAD/dataset/Oulu_NPU",
    # "/datasets/newout/PAD/dataset/Replay-Attack",
    # "/datasets/newout/PAD/dataset/Replay-Mobile",
    # "/datasets/newout/PAD/dataset/Rose",
    # "/datasets/newout/PAD/dataset/SiW",
    # "/datasets/newout/PAD/dataset/SiW-Mv2",
    # "/datasets/newout/PAD/dataset/SuHiFiMask",
    # "/datasets/newout/PAD/dataset/SWAX",

    # "/datasets/newout/from_nizar1/alive-images_1M",
    # "/datasets/newout/from_nizar1/alive_2025-04-17",
    # "/datasets/newout/from_nizar1/2025-05-16_web_alive-test",
    # "/datasets/newout/from_nizar1/2025-04-23/real",
    # "/datasets/newout/from_nizar1/2025_05_15_deep_live_cam/real",
    # "/datasets/newout/from_nizar1/2025_05_16_deep_live_cam/real",
    # "/datasets/newout/from_nizar/alive_images",

    # "/datasets/newout/from_nizar",
    # "/datasets/newout/from_nizar1",
    # "/datasets/newout/1m_faces_91__99",
    # "/datasets/newout/add_free/70k-face-frames",
    # "/datasets/newout/add_free/deepf_real_images",
    # "/datasets/newout/add_free/DeepfDatabase",
    # "/datasets/newout/add_free/deepfdatatrainvalid",
    # "/datasets/newout/add_free/DeepFImages",
    # "/datasets/newout/add_free/DFFMD",
    # "/datasets/newout/add_free/RealvsFfaces",
    # "/datasets/newout/add_free/styleGAN3_annotated",
    # "/datasets/newout/AgedSyntheticImages",
    # "/datasets/newout/CelebDF_v2",
    # "/datasets/newout/DeeperForensics",
    # "/datasets/newout/DEEPFAKE_CHALLENGE",
    # "/datasets/newout/DeepFakeFace",
    # "/datasets/newout/DeepfakeTIMIT",
    # "/datasets/newout/DFFD",
    # "/datasets/newout/DFGC-2021",
    # # "/datasets/newout/DFGC-2022",
    # "/datasets/newout/DFMNIST+",
    # "/datasets/newout/FF++_HifiFace",
    # "/datasets/newout/how_fmc",
    # "/datasets/newout/iFakeFaceDB",
    # "/datasets/newout/MegaFS",
    # "/datasets/newout/new_df",

    "/datasets/work/vLLM/data/fmt_error_mis_4+13fmt_0",
]
@dataclass
class Sample:
    rec_id: str
    image_path: Path
    base_prompt: str
    cls_label: int


class ImagesDataset(Dataset):

    def __init__(
        self,
        input_path: str,
        image_root: str = "",
        prompt_list_path: str = "playground/prompts.txt",
        strict_exists: bool = False,
        which_part: int = 0,
    ):
        super(ImagesDataset, self).__init__()
        self.items: List[Sample] = []
        self.strict_exists = strict_exists

        full_file_paths = []
        for in_path in input_paths:
            paths = get_filepaths(in_path)
            full_file_paths += paths

        n_pad = 0
        n_df = 0
        n_real = 0
        n_makeup = 0
        n_total = 0
        pad_paths = []
        df_paths = []
        real_paths = []
        makeup_paths = []
        for img_path in full_file_paths:
            n_total += 1
            
            label = get_label_all(img_path)

            # if label == 0 and "Replay-Mobile" in img_path:
            # if label == 0 and "from_nizar" in img_path:
            if label == 0:
                n_real += 1
                real_paths.append(img_path)
            elif label == 1:
                n_pad += 1
                pad_paths.append(img_path)
            elif label == 2:
                n_df += 1
                df_paths.append(img_path)
            elif label == 3:
                n_makeup += 1
                makeup_paths.append(img_path)

        # full_file_paths = pad_paths + df_paths + makeup_paths + real_paths
        full_file_paths = real_paths
        # full_file_paths = random.sample(full_file_paths, len(full_file_paths))
        full_file_paths = sorted(full_file_paths)
        print(f"n_real:{n_real}, n_pad:{n_pad}, n_df:{n_df}, n_makeup:{n_makeup}")

        if which_part != 0:
            data_len = len(full_file_paths)
            seg_len = data_len / 7
            start_id = int((which_part - 1) * seg_len)
            end_id = int(which_part * seg_len)
            if end_id > data_len - 1:
                end_id = data_len - 1
            full_file_paths = full_file_paths[start_id:end_id]
            # full_file_paths = random.sample(full_file_paths, int(len(full_file_paths)/4))

        image_root_path = Path(image_root) if image_root else Path(".")
        prompts = read_txt_file(prompt_list_path)
        if not prompts:
            prompts = ['The image is a human face image. Is it real or fake? Why?']

        rid = 50000000
        for img_path in full_file_paths:
            head, fname = ntpath.split(img_path)
            if os.path.isdir(img_path) is False and (
                fname.endswith(".jpg") or fname.endswith(".jpeg")
            ):
                rec_id = rid
                rid += 1
                img_path = img_path.replace("\\", "/")
                base_prompt = random.choice(prompts)

                label = get_label_all(img_path)

                if label == 0:
                    cls_label = 0
                elif label >= 1 and label <= 3:
                    cls_label = 1
                else:
                    continue

                self.items.append(
                    Sample(
                        rec_id=rec_id,
                        image_path=img_path,
                        base_prompt=base_prompt,
                        cls_label=cls_label,
                    )
                )

        # keep something for compatibility; used in main but not critical
        self._raw_records = self.items

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
    return {
        "rec_ids": rec_ids,
        "paths": paths,
        "images": images,
        "base_prompts": base_prompts,
        "cls_labels": cls_labels,
    }


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
            [x.squeeze(0) for x in ids],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
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

    assert torch.cuda.is_available(), "CUDA is required for GPU batching."

    device_id = int(args.device)

    if args.which_part != 0:
        device_id = int(args.which_part)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(device_id))
    device = torch.device(f"cuda:{device_id}")

    llava_groups = {
        "mistral": "checkpoints/effaa-llava-mistral-7b-lora_1",
        "phi": "checkpoints/ffaa-phi-3-mini",
    }
    # model_path = llava_groups["mistral"]
    model_path = args.model_path

    # ----- load LLaVA -----
    model, image_processor, tokenizer = load_llava(model_path, device_id)
    model = model.to(device)

    # ----- load MIDS + tokenizer/processor -----
    mids_path = args.mids_path or os.path.join(model_path, "mids.pth")
    t5_tokenizer = AutoTokenizer.from_pretrained(
        "models/t5-base", use_fast=False, legacy=False
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "models/clip-vit-large-patch14-336"
    )
    mids_model = load_mids(mids_path, device_id)
    mids_model.eval()

    dataset = ImagesDataset(
        input_path=args.input_dir,
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

    total = len(dataset)
    seen = 0
    n_failed = 0
    n_miss = 0
    n_lowp = 0

    for batch in loader:
        rec_ids: List[str] = batch["rec_ids"]
        paths: List[str] = batch["paths"]
        images: List[Any] = batch["images"]
        base_prompts: List[str] = batch["base_prompts"]
        cls_labels: List[int] = batch["cls_labels"]

        t0 = time.time()

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

        for idx, (rid, pth, cls_label, answers3) in enumerate(
            zip(rec_ids, paths, cls_labels, answers_3_per_image)
        ):
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
                    answer_result = answer_json["Analysis result"].lower()
                    new_answer_json = {
                        "Image description": answer_json["Image description"],
                        "Forgery reasoning": answer_json["Forgery reasoning"],
                    }
                    new_answer = "\n".join(
                        f"{key}: {value}" for key, value in new_answer_json.items()
                    )
                    answers_result.append(answer_result)
                    processed_answers.append(new_answer)
                    if answer_result == "real":
                        cls_out = 0
                    else:
                        cls_out = 1
                    cls_outputs.append(cls_out)

            if all_ok is False:
                head, fname = ntpath.split(pth)
                err_path = r'/datasets/work/vLLM/data/fmt_error_mis_4+13fmt'
                label = get_label_all(pth)
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

            # ----- MIDS selection for this image -----
            image_0 = cv2.imread(pth, cv2.IMREAD_COLOR) #zzzzzz for compatibility with train_mids.py
            image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
            image_0 = Image.fromarray(image_0)
            mids_info = run_mids_for_one(
                mids_model=mids_model,
                clip_processor=clip_processor,
                t5_tokenizer=t5_tokenizer,
                image=image_0,
                answers=answers3,
                device=device,
            )

            if mids_info is not None:
                best_answer_json, _ = decode_response(answers3[mids_info["best_answer_idx"]])
                match_score = mids_info["match_score"]
                label = get_label_all(pth)
                if (
                    (best_answer_json['Analysis result'].lower() == 'real' and cls_label == 1)
                    or (best_answer_json['Analysis result'].lower() == 'fake' and cls_label == 0)
                ):
                    head, fname = ntpath.split(pth)
                    err_path = r"/datasets/work/vLLM/data/mis-classified_4+13fmt_allreal"
                    if cls_label == 0 and label == 0:
                        dst_path = os.path.join(err_path, "real")
                    elif cls_label == 1 and label == 1:
                        dst_path = os.path.join(err_path, "fake", "pad")
                    elif cls_label == 1 and label == 2:
                        dst_path = os.path.join(err_path, "fake", "deepfake")
                    elif cls_label == 1 and label == 3:
                        dst_path = os.path.join(err_path, "fake", "makeup")
                    else:
                        continue

                    if not os.path.exists(dst_path):
                        os.makedirs(dst_path, exist_ok=True)
                    dst_path = os.path.join(dst_path, fname)
                    dst_path = unique_filename_by_size(dst_path, pth)
                    copyfile(pth, dst_path)
                    out_item = build_mids_out_item(
                        rec_id=rid,
                        image_path=dst_path,
                        cls_label=cls_label,
                        processed_answers=processed_answers,
                        answers_result=answers_result,
                        cls_outputs=cls_outputs,
                    )
                    write_mids_out_item_json(dst_path, out_item)
                    print((f"Missed: {pth}"))
                    n_miss += 1
                elif (
                    cls_label == 0
                    and best_answer_json['Analysis result'].lower() == 'real'
                    and match_score < 0.99
                ):
                    head, fname = ntpath.split(pth)
                    dst_path = r"/datasets/work/vLLM/data/lowp_real_4+13fmt/real"
                    if not os.path.exists(dst_path):
                        os.makedirs(dst_path, exist_ok=True)
                    dst_path = os.path.join(dst_path, fname)
                    dst_path = unique_filename_by_size(dst_path, pth)
                    copyfile(pth, dst_path)
                    out_item = build_mids_out_item(
                        rec_id=rid,
                        image_path=dst_path,
                        cls_label=cls_label,
                        processed_answers=processed_answers,
                        answers_result=answers_result,
                        cls_outputs=cls_outputs,
                    )
                    write_mids_out_item_json(dst_path, out_item)
                    print((f"Lowp: {pth}"))
                    n_lowp += 1

        seen += len(rec_ids)
        print(
            f"[{args.which_part}/{seen}/{total}] "
            f"{100.0*seen/total:.1f}% done, time:{time.time() - t0:.1f}s"
        )

    print(f"[failed:{n_failed}/{total}],[n_miss:{n_miss}, n_lowp:{n_lowp}]")


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch LLaVA inference with conditional prompts per image (GPU) + MIDS selection."
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/effaa-llava-mistral-7b-lora_4",
        help="Path to LLaVA model.",
    )
    p.add_argument(
        "--mids_path",
        type=str,
        default="checkpoints/effaa-llava-mistral-7b-lora_4/mids.pth",
        help="Path to MIDS checkpoint (.pth). "
             "Defaults to <model_path>/mids.pth if not set.",
    )
    p.add_argument(
        "--which_part",
        type=int,
        default=0,
        help="will use 1~7gpu and the corresponding part of dataset."
             "if 0: all. else 1~7: the corresponding part",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default="/datasets/work/vLLM/new_mytest",
        help="Path to the input directory containing images.",
    )
    p.add_argument(
        "--image_root",
        type=str,
        default="/datasets/newout",
        help="Optional root to prefix each 'image' path.",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="/datasets/newout/vqa_info/temp/mids_dir.json",
        help="Where to write the augmented JSON.",
    )
    p.add_argument(
        "--prompt_list",
        type=str,
        default="playground/prompts.txt",
        help="Path to a newline-separated list of prompts.",
    )
    p.add_argument(
        "--strict_exists",
        action="store_true",
        help="Skip records whose image files are missing.",
    )

    # batching / performance
    p.add_argument("--batch_size", type=int, default=160, help="Images per batch.")
    p.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="DataLoader workers for image I/O.",
    )
    p.add_argument(
        "--device", type=int, default=0, help="CUDA device id (e.g., 0, 1, ...)."
    )

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
