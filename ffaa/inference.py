import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import dlib
import cv2
import numpy as np

DETECTOR = dlib.get_frontal_face_detector()
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
    image = np.array(image)
    faces = DETECTOR(image, 1)
    if len(faces) == 0:
        return None
    
    d = faces[0]

    img_height, img_width = image.shape[:2]

    # crop face
    left, top, right, bottom = d.left(), d.top(), d.right(), d.bottom()

    # add padding
    max_padding_top = top
    max_padding_bottom = img_height - bottom
    max_padding_left = left
    max_padding_right = img_width - right

    padding = min(max_padding_top, max_padding_bottom, max_padding_left, max_padding_right, padding)
    face_img = image[top-padding:bottom+padding, left-padding:right+padding]

    face_img_resized = cv2.resize(face_img, (336, 336))
    face_img_resized = Image.fromarray(face_img_resized)
    face_img_resized.save(TMP_IMG_PATH)
    return face_img_resized


def load_llava(model_path, device_id):
    device_map = device_id
    # device_map = "auto"
    kwargs = {"device_map": device_map, 'torch_dtype': torch.float16,
              # "load_in_4bit": True,
              "use_flash_attention_2": True
              }

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
    # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4",
    #                                   bnb_4bit_use_double_quant=True)
    # model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, quantization_config=quant_config,
    #                                               trust_remote_code=True, )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign = True)
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

# model path
# llava_groups = {
#     'mistral': "checkpoints/ffaa-mistral-7b",
#     'phi': "checkpoints/ffaa-phi-3-mini",
# }
# # mids path
# mids_path = f"checkpoints/ffaa-mistral-7b/mids.pth"
llava_groups = {
    'mistral': "checkpoints/effaa-llava-mistral-7b-lora_3",
    'phi': "checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_0",
}
# mids path
mids_path = f"checkpoints/effaa-llava-mistral-7b-lora_3/mids.pth"

device_id = 0

# load mllm
mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["mistral"], device_id)

# load mids
t5_tokenizer = AutoTokenizer.from_pretrained('models/t5-base', use_fast=False, legacy=False)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336")
mids = load_mids(mids_path, device_id)

def inference(args):
    # args
    image_path = args.image_path
    crop = args.crop

    mids.eval()

    start_t = time.time()

    print(f'USER: {args.prompt}\n')
    device = torch.device(f'cuda:{device_id}')
    def run_mistral(image, answer_holder):
        with torch.no_grad():
            answers = get_llava_answer(mistral_model, mistral_tokenizer, mistral_image_processor,
                                    image, args.prompt, args.temperature, args.top_p, args.num_beams,
                                    args.max_new_tokens, args.generate_num['mistral'], 'v1')
            answer_holder.extend(answers)
    
    image = load_image(image_path)
    if crop == 1:
        image = crop_face(image)
        if image is None:
            print('No face detected')
            return
    else:
        image.save(TMP_IMG_PATH)

    threads = []

    mistral_answer_holder = []

    mistral_thread = threading.Thread(target=run_mistral, args=(image, mistral_answer_holder))
    threads.append(mistral_thread)
    mistral_thread.start()

    # wait all threads complete
    for thread in threads:
        thread.join()

    answers = mistral_answer_holder
    scores = []

    # mask answers
    answers_result = []
    processed_answers = []
    for answer in answers:
        answer, answer_res = mask_result(answer)
        answers_result.append(answer_res)
        processed_answers.append(answer)
    
    # easy or hard
    if len(set(answers_result)) == 1:
        qs_difficulty = 'easy'
    else:
        qs_difficulty = 'hard'
    
    if len(answers) == 3:
        N=1;M=1
    elif len(answers) == 2:
        N=0;M=1
    elif len(answers) == 1:
        N=0;M=0
    
    mids_s = time.time()
    # mids
    with torch.inference_mode():
        input_image = clip_processor(images=image, return_tensors='pt')['pixel_values']
        answer_ids = t5_tokenizer(processed_answers, return_tensors="pt", padding="longest", max_length=t5_tokenizer.model_max_length, truncation=True)
        logits = mids(answer_ids.to(device), input_image.to(device), None, 1, N,M)['logits']
        scores = F.softmax(logits, dim=2).squeeze(0)
        best_answer_idx, pred, match_score, forgery_score = make_decision(answers_result, scores)
    

    best_answer_json, _ = decode_response(answers[best_answer_idx])
    # CLS is used to visualize heatmap related to the final classification result
    if best_answer_json['Analysis result'].lower() == 'real':
        best_answer_cls = 0
        if float(best_answer_json['Probability']) < 0.8 and random.random() < 0.5:
            best_answer_json['Forgery reasoning'] = best_answer_json['Forgery reasoning'] + f" It's close to '"'real'"', but not completely certain."
            best_answer_json['Analysis result'] = "ambiguous"
        elif float(best_answer_json['Probability']) < 0.8 and random.random() >= 0.5:
            best_answer_json['Forgery reasoning'] = best_answer_json['Forgery reasoning'] + f" It's close to '"'real'"', but it's hard to say with complete certainty."
            best_answer_json['Analysis result'] = "ambiguous"
    else:
        best_answer_cls = 3
    orginal_answer = answers[best_answer_idx]
    best_answer_json['Match score'] = f"{match_score:.4f}"
    best_answer_json['Difficulty'] = qs_difficulty

    best_answer = answer_format(best_answer_json)

    end_t = time.time()
    print(f"total time : {end_t-start_t}s, MIDS : {end_t-mids_s}")

    return best_answer, best_answer_cls, orginal_answer

def main(image_path):
    """
        CUDA_VISIBLE_DEVICES=0 python inference.py --crop 1 --visualize 1
    """

    prompt_list = read_txt_file("playground/prompts.txt")
    # prompt = 'The image is a human face image. Is it real or fake? Why?'
    prompt = random.choice(prompt_list)

    crop = 0
    visualize = 0

    genarate_num_dict = {
        'mistral': 3,
        'phi': 0,
    }

    args = type('Args', (), {
        "device": 0,
        "image_path": image_path,
        "prompt": prompt,
        "crop": crop,
        "conv_mode": None,
        "llava_groups": llava_groups,
        "mids_path": mids_path,
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "generate_num": genarate_num_dict,
        "max_new_tokens": 512
    })()

    answer, cls, org_answer = inference(args)

    if visualize == 1:
        print('Visualize heatmaps...')
        from visualize import get_heatmap
        visualize_args = {
            'checkpoint': mids_path,
            'clip_processor': CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336"),
            'savename': TMP_IMG_PATH.split('.')[0],
            'dir': 'heatmaps'
        }
        get_heatmap(TMP_IMG_PATH, org_answer, cls, layer=1, **visualize_args)
        get_heatmap(TMP_IMG_PATH, org_answer, cls, layer=2, **visualize_args)

    return answer

if __name__ == "__main__":
    input_path = "./playground/mytest"
    full_file_paths = get_filepaths(input_path)
    num = 0
    for path in full_file_paths:
        if ".jpg" in path or ".jpeg" in path:
            print(path)
            answer = main(path)
            print(f'FFAA: {answer}\n')
