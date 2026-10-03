import time
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # PAAS: respect a device chosen before import
from transformers import AutoTokenizer, CLIPProcessor
from transformers import BitsAndBytesConfig
import transformers
transformers.logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model import *
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from mids.selector import make_decision
from utils.file_utils import *
from utils.llava_utils import *
from utils.mids_utils import *
import threading
import argparse
import random
from typing import Any, List, Optional

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#import dlib
import cv2
import numpy as np

#DETECTOR = dlib.get_frontal_face_detector()
TMP_IMG_PATH = 'test.png'

def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)  # Self-explanatory.

def crop_face(image, padding=40):
#    image = np.array(image)
#   faces = DETECTOR(image, 1)
#    if len(faces) == 0:
#        return None
#    
#    d = faces[0]
#
#    img_height, img_width = image.shape[:2]
#
#    # crop face
#    left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()
#
#    # add padding
#    max_padding_top = top
#    max_padding_bottom = img_height - bottom
#    max_padding_left = left
#    max_padding_right = img_width - right
#
#    padding = min(max_padding_top, max_padding_bottom, max_padding_left, max_padding_right, padding)
#    face_img = image[top-padding:bottom+padding, left-padding:right+padding]
#
#    face_img_resized = cv2.resize(face_img, (336, 336))
#    face_img_resized = Image.fromarray(face_img_resized)
#    face_img_resized.save(TMP_IMG_PATH)
#   return face_img_resized
    return []


def load_llava(model_path, device_id):
    device_map = device_id
    # device_map = "auto"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        # bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    kwargs = {"device_map": device_map, 'torch_dtype': torch.float16,
              # "load_in_8bit": True,
              # "load_in_4bit": True,
              # "quantization_config": quant_config,
              "use_flash_attention_2": True
              }

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign=True)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor

    return model, image_processor, tokenizer

def load_mids(model_path, device_id):
    from mids.mids_arch import MIDS
    model = MIDS()

    model_state_dict = model.state_dict()

    finetuned_state_dict = torch.load(model_path)
    finetuned_state_dict = {key.replace('module.', ''): value for key, value in finetuned_state_dict.items()}

    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)

    return model.to(dtype=torch.float32, device=torch.device(f'cuda:{device_id}'))

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
            # prompt = prompt + "\n" + prompt # zzzzzz
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


def _build_llava_batch_input_ids(model, tokenizer, qs_list, conv_mode):
    ids = []
    for qs in qs_list:
        prompt = get_llava_prompt(model, qs, conv_mode)
        ids.append(
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).squeeze(0)
        )

    input_ids = torch.nn.utils.rnn.pad_sequence(
        ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    return input_ids.to(model.device)


@torch.inference_mode()
def get_llava_answer_batch(
    model,
    tokenizer,
    image_processor,
    images: List[Any],
    prompts: List[str],
    temperature,
    top_p,
    num_beams,
    max_new_tokens,
    per_sample_generate_num,
    conv_mode='v1',
):
    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same length")
    if not images:
        return []

    image_sizes = [image.size for image in images]
    image_tensor = process_images(
        images,
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)

    batch_size = len(images)
    outputs = [[] for _ in range(batch_size)]
    condition_prompt = 'This is a _ human face. What evidence do you have?'

    input_ids = _build_llava_batch_input_ids(model, tokenizer, prompts, conv_mode)
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
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text.strip())

    if per_sample_generate_num <= 1:
        return outputs

    prompts_2 = []
    prompts_3 = []
    for idx in range(batch_size):
        try:
            response_json, _ = decode_response(outputs[idx][0])
            answer_result = response_json['Analysis result'].lower()
        except Exception:
            answer_result = 'fake'

        if answer_result == 'real':
            prompts_2.append(condition_prompt.replace('_', 'fake'))
            prompts_3.append(condition_prompt.replace('_', 'real'))
        else:
            prompts_2.append(condition_prompt.replace('_', 'real'))
            prompts_3.append(condition_prompt.replace('_', 'fake'))

    input_ids = _build_llava_batch_input_ids(model, tokenizer, prompts_2, conv_mode)
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
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text.strip())

    if per_sample_generate_num <= 2:
        return outputs

    input_ids = _build_llava_batch_input_ids(model, tokenizer, prompts_3, conv_mode)
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
        attention_mask=(input_ids != tokenizer.pad_token_id).long(),
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text.strip())

    return outputs
