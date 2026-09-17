import ntpath
import random
import string
import json
from datetime import datetime
from io import BytesIO
from typing import List

import cv2
import base64
import os
from PIL import Image, UnidentifiedImageError
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import torch
# torch.set_flush_denormal(True)
from models import *
from transformers import AutoTokenizer, CLIPProcessor

from flask import (Flask, render_template, request, jsonify, g)
from werkzeug.utils import secure_filename

# from insightface.app import FaceAnalysis
from yolo11_cls_onnx import Yolo11ClsONNX
from flask_cors import CORS
from mids.selector import make_decision_batch

# # Initialize globally (do this once at startup)
# app = FaceAnalysis(name="buffalo_l")
# app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for first GPU

det_model_path = "./rot_det_model/yolo11n-rotation2/weights/best.onnx"
det_model = Yolo11ClsONNX(
    onnx_path=det_model_path,
    imgsz=224,
    class_names=["0", "180", "270", "90"],
)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = APP_ROOT + "/images"
DEFAULT_LIVENESS_PROMPT = "The image is a human face image. Is it real or fake? Why?"
IMAGE_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
}

# model path
# llava_groups = {
#     'mistral': "checkpoints/ffaa-mistral-7b",
#     'phi': "checkpoints/ffaa-phi-3-mini",
# }
# # mids path
# mids_path = f"checkpoints/ffaa-mistral-7b/mids.pth"
llava_groups = {
    # 'mistral': "checkpoints/effaa-llava-mistral-7b-lora_6",
    # 'mistral': "checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_2",
    # 'mistral': "checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_1",
    'mistral': "checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0",
    'phi': "checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_3",
}
# mids path
# mids_path = f"checkpoints/effaa-llava-mistral-7b-lora_6/mids.pth"
# mids_path = f"checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_2/mids.pth"
# mids_path = f"checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_1/mids.pth"
mids_path = f"checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0/mids.pth"

g_crop = 0
device_id = 0

# load mllm
mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["mistral"], device_id)
# mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["phi"], device_id)

# load mids
t5_tokenizer = AutoTokenizer.from_pretrained('models/t5-base', use_fast=False, legacy=False)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336")
mids = load_mids(mids_path, device_id)


app = Flask(__name__)
CORS(app) # Enable CORS for all routes and origins
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '1024')) * 1024 * 1024


# force browser to hold no cache. Otherwise old result might return.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def homepage():
    # return render_template('home.html')
    return render_template('home_two.html')


def getRotatedAngle(img):

    rot_angle, conf = det_model.predict(img)  # angle:(int 0,90,180,270) conf:(float)

    return rot_angle


def upload_request_images_face(content, subdir1, subdir2):
    """ Upload request image to folder """
    file_name = secure_filename(content.filename)

    file_name_without_ext, ext = os.path.splitext(file_name)
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext

    subdir = get_request_image_subdir()

    # subdir = os.path.join(subdir, subdir2)
    # if not os.path.exists(subdir):
    #     os.makedirs(subdir)

    file_path = os.path.join(subdir, fname)
    content.save(file_path)

    return file_path


def get_request_image_subdir():
    ipaddr = request.headers.get("X-Forwarded-For", request.remote_addr)
    if ipaddr:
        ipaddr = ipaddr.split(",")[0].strip()
    if not ipaddr:
        ipaddr = "unknown"

    subdir = os.path.join('./images', ipaddr)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def get_image_suffix_from_bytes(image_bytes):
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Failed to decode image") from exc

    return IMAGE_FORMAT_TO_EXT.get(image_format, ".png")


def save_request_image_bytes(image_bytes, suffix):
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix
    file_path = os.path.join(get_request_image_subdir(), fname)
    with open(file_path, "wb") as file_obj:
        file_obj.write(image_bytes)
    return file_path


def get_random_string():
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(8))
    return result_str


def get_liveness_prompt():
    prompt_list = read_txt_file("playground/prompts.txt")
    if len(prompt_list) == 0:
        return DEFAULT_LIVENESS_PROMPT
    return prompt_list[0]


def build_face_liveness_string_response(answer):
    output = get_jsonfmt(answer)
    json_output = json.dumps(output, indent=2)
    print(json_output)
    return {
        'success': True,
        'face_liveness': json_output,
    }


def build_face_liveness_object_response(answer):
    output = get_jsonfmt(answer)
    json_output = json.dumps(output, indent=2)
    print(json_output)
    return {
        'success': True,
        'face_liveness': output,
    }


def finalize_best_answer(answers, answers_result, best_answer_idx, match_score):
    if len(set(answers_result)) == 1:
        qs_difficulty = 'easy'
    else:
        qs_difficulty = 'hard'

    best_answer_json, _ = decode_response(answers[best_answer_idx])
    if best_answer_json['Analysis result'].lower() == 'real':
        best_answer_cls = 0
        if float(best_answer_json['Probability']) < 0.8 or (match_score < 0.99 and match_score > 0.8):
            if qs_difficulty == 'hard':
                idx = len(answers_result) - 1
                for ans_res in reversed(answers_result):
                    if ans_res == 'fake':
                        break
                    idx -= 1
                ans_sel_json, _ = decode_response(answers[idx])
                best_answer_json['Forgery type'] = ans_sel_json['Forgery type']
                best_answer_json['Analysis result'] = 'ambiguous'
                best_answer_json['Forgery reasoning'] = (
                    ans_sel_json['Forgery reasoning'] + ' It\'s close to "real", but not completely certain.'
                )
            else:
                best_answer_json['Analysis result'] = "ambiguous"
                best_answer_json['Forgery reasoning'] = (
                    best_answer_json['Forgery reasoning'] + ' It\'s close to "real", but not completely certain.'
                )
        elif match_score <= 0.8:
            best_answer_json['Forgery type'] = 'ambiguous'
            best_answer_json['Analysis result'] = 'likely_fake'
            if qs_difficulty == 'hard':
                idx = len(answers_result) - 1
                for ans_res in reversed(answers_result):
                    if ans_res == 'fake':
                        break
                    idx -= 1
                ans_sel_json, _ = decode_response(answers[idx])
                best_answer_json['Forgery type'] = ans_sel_json['Forgery type']
                best_answer_json['Forgery reasoning'] = (
                    ans_sel_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
                )
            else:
                best_answer_json['Forgery reasoning'] = (
                    best_answer_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
                )
    else:
        best_answer_cls = 3

    # if best_answer_json['Analysis result'].lower() == 'fake' and qs_difficulty == 'easy' and match_score < 0.2:
    #     best_answer_json['Analysis result'] = 'real'
    #     best_answer_json['Forgery type'] = 'None'
    #     best_answer_json['Forgery reasoning'] = (
    #         best_answer_json['Forgery reasoning'] + ' It\'s closer to "real", but not completely certain.'
    #     )
    # elif best_answer_json['Analysis result'].lower() == 'real' and qs_difficulty == 'easy' and match_score < 0.2:
    #     best_answer_json['Analysis result'] = 'fake'
    #     best_answer_json['Forgery type'] = 'None'
    #     best_answer_json['Forgery reasoning'] = (
    #         best_answer_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
    #     )

    best_answer_json['Match score'] = f"{match_score:.4f}"
    best_answer_json['Difficulty'] = qs_difficulty

    best_answer = answer_format(best_answer_json)
    return best_answer, best_answer_cls


def check_live_batch(image_paths: List[str]):
    prompt = get_liveness_prompt()

    crop = g_crop
    genarate_num_dict = {
        'mistral': 3,
        'phi': 0,
    }
    device = torch.device(f'cuda:{device_id}')

    results = [None] * len(image_paths)
    valid_indices = []
    valid_paths = []
    valid_images = []

    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        angle = getRotatedAngle(img)
        if angle > 0:
            results[idx] = {'error': 'The image appears to be rotated. Please try again with a straightened image.'}
            print("error: The image appears to be rotated. Please try again with a straightened image.")
            continue

        image = load_image(image_path)
        if crop == 1:
            image = crop_face(image)
            if image is None:
                results[idx] = {'error': 'No face detected'}
                print("error: No face detected")
                continue

        valid_indices.append(idx)
        valid_paths.append(image_path)
        valid_images.append(image)

    if not valid_images:
        return results

    args = type('Args', (), {
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        "generate_num": genarate_num_dict,
    })()

    start_t = time.time()
    with torch.no_grad():
        answers_3_per_image = get_llava_answer_batch(
            mistral_model,
            mistral_tokenizer,
            mistral_image_processor,
            valid_images,
            [prompt] * len(valid_images),
            args.temperature,
            args.top_p,
            args.num_beams,
            args.max_new_tokens,
            args.generate_num['mistral'],
            'v1',
        )

    flattened_processed_answers = []
    flattened_answers_result = []
    mids_indices = []
    mids_paths = []
    mids_images = []
    mids_answers_per_image = []
    answers_result_per_image = []

    for batch_idx, answers in enumerate(answers_3_per_image):
        answers_result = []
        processed_answers = []
        try:
            for answer in answers:
                masked_answer, answer_res = mask_result(answer)
                answers_result.append(answer_res)
                processed_answers.append(masked_answer)
        except Exception as exc:
            original_idx = valid_indices[batch_idx]
            results[original_idx] = {'error': str(exc)}
            print(f"error: {exc}")
            continue

        mids_indices.append(valid_indices[batch_idx])
        mids_paths.append(valid_paths[batch_idx])
        mids_images.append(valid_images[batch_idx])
        mids_answers_per_image.append(answers)
        answers_result_per_image.append(answers_result)
        flattened_answers_result.extend(answers_result)
        flattened_processed_answers.extend(processed_answers)

    if not mids_images:
        return results

    mids_s = time.time()
    with torch.inference_mode():
        input_images = clip_processor(images=mids_images, return_tensors='pt')['pixel_values']
        answer_ids = t5_tokenizer(
            flattened_processed_answers,
            return_tensors="pt",
            padding="longest",
            max_length=t5_tokenizer.model_max_length,
            truncation=True,
        )
        logits = mids(
            answer_ids.to(device),
            input_images.to(device),
            None,
            len(mids_images),
            1,
            1,
        )['logits']
        scores = F.softmax(logits, dim=2)
        best_answer_idxs, preds, match_scores, forgery_scores = make_decision_batch(
            flattened_answers_result,
            scores,
        )

    for batch_idx, original_idx in enumerate(mids_indices):
        answers = mids_answers_per_image[batch_idx]
        answers_result = answers_result_per_image[batch_idx]
        best_answer_idx = best_answer_idxs[batch_idx]
        match_score = match_scores[batch_idx]

        try:
            best_answer, cls = finalize_best_answer(
                answers,
                answers_result,
                best_answer_idx,
                match_score,
            )
        except Exception as exc:
            results[original_idx] = {'error': str(exc)}
            print(f"error: {exc}")
            continue

        head, fname = ntpath.split(mids_paths[batch_idx])
        file_name_without_ext, ext = os.path.splitext(fname)
        text_path = os.path.join(head, file_name_without_ext + '.txt')
        with open(text_path, "w") as f:
            f.write(best_answer)

        results[original_idx] = {'success': True, 'face_liveness': best_answer}

    end_t = time.time()
    print(f"batch total time : {end_t - start_t}s, MIDS : {end_t - mids_s}")

    return results


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
        N = 1;
        M = 1
    elif len(answers) == 2:
        N = 0;
        M = 1
    elif len(answers) == 1:
        N = 0;
        M = 0

    mids_s = time.time()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR) #zzzzzz for compatibility with train_mids.py
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # mids
    with torch.inference_mode():
        input_image = clip_processor(images=image, return_tensors='pt')['pixel_values']
        answer_ids = t5_tokenizer(processed_answers, return_tensors="pt", padding="longest",
                                  max_length=t5_tokenizer.model_max_length, truncation=True)
        logits = mids(answer_ids.to(device), input_image.to(device), None, 1, N, M)['logits']
        scores = F.softmax(logits, dim=2).squeeze(0)
        best_answer_idx, pred, match_score, forgery_score = make_decision(answers_result, scores)

    best_answer_json, _ = decode_response(answers[best_answer_idx])
    # CLS is used to visualize heatmap related to the final classification result
    if best_answer_json['Analysis result'].lower() == 'real':
        best_answer_cls = 0
        if float(best_answer_json['Probability']) < 0.8 or (match_score < 0.99 and match_score > 0.8):
        # if float(best_answer_json['Probability']) < 0.8 or (match_score < 0.9 and match_score > 0.8):
            # best_answer_json['Analysis result'] = "ambiguous"
            if qs_difficulty == 'hard':
                idx = len(answers_result) - 1
                for ans_res in reversed(answers_result):
                    if ans_res == 'fake':
                        break
                    idx -= 1
                ans_sel_json, _ = decode_response(answers[idx])
                best_answer_json['Forgery type'] = ans_sel_json['Forgery type']
                best_answer_json['Analysis result'] = 'ambiguous'#ans_sel_json['Analysis result']
                best_answer_json['Forgery reasoning'] = (
                    ans_sel_json['Forgery reasoning'] + ' It\'s close to "real", but not completely certain.'
                )
            else:
                best_answer_json['Analysis result'] = "ambiguous"
                best_answer_json['Forgery reasoning'] = (
                    best_answer_json['Forgery reasoning'] + ' It\'s close to "real", but not completely certain.'
                )
        elif match_score <= 0.8:
            best_answer_json['Forgery type'] = 'ambiguous'
            best_answer_json['Analysis result'] = 'likely_fake'
            if qs_difficulty == 'hard':
                idx = len(answers_result) - 1
                for ans_res in reversed(answers_result):
                    if ans_res == 'fake':
                        break
                    idx -= 1
                ans_sel_json, _ = decode_response(answers[idx])
                best_answer_json['Forgery type'] = ans_sel_json['Forgery type']
                best_answer_json['Forgery reasoning'] = (
                    ans_sel_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
                )
            else:
                best_answer_json['Forgery reasoning'] = (
                    best_answer_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
                )
    else:
        best_answer_cls = 3

    # if best_answer_json['Analysis result'].lower() == 'fake' and qs_difficulty == 'easy' and match_score < 0.2:
    #     best_answer_json['Analysis result'] = 'real'
    #     best_answer_json['Forgery type'] = 'None'
    #     best_answer_json['Forgery reasoning'] = (
    #         best_answer_json['Forgery reasoning'] + ' It\'s closer to "real", but not completely certain.'
    #     )    
    # elif best_answer_json['Analysis result'].lower() == 'real' and qs_difficulty == 'easy' and match_score < 0.2:
    #     best_answer_json['Analysis result'] = 'fake'
    #     best_answer_json['Forgery type'] = 'None'
    #     best_answer_json['Forgery reasoning'] = (
    #         best_answer_json['Forgery reasoning'] + ' It\'s closer to "fake", but not completely certain.'
    #     )

    orginal_answer = answers[best_answer_idx]
    best_answer_json['Match score'] = f"{match_score:.4f}"
    best_answer_json['Difficulty'] = qs_difficulty

    # if best_answer_cls == 0:
    #     best_answer_json = fix_blured_real(best_answer_json, answers, answers_result)

    best_answer = answer_format(best_answer_json)

    end_t = time.time()
    print(f"total time : {end_t - start_t}s, MIDS : {end_t - mids_s}")

    return best_answer, best_answer_cls, orginal_answer


def check_live(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    angle = getRotatedAngle(img)
    if angle > 0:
        res = {'error': 'The image appears to be rotated. Please try again with a straightened image.'}
        # res = {'success': False, 'face_liveness': 'The image appears to be rotated. Please try again with a straightened image.'}
        print(f"error: The image appears to be rotated. Please try again with a straightened image.")
        return res

    # select the image and write your prompt here
    prompt = get_liveness_prompt()

    crop = g_crop
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

    # print(answer)

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

    res = {'success': True, 'face_liveness': answer}

    head, fname = ntpath.split(image_path)
    file_name_without_ext, ext = os.path.splitext(fname)
    text_path = os.path.join(head, file_name_without_ext+'.txt')
    with open(text_path, "w") as f:
        f.write(answer)

    return res


@app.route('/face_liveness', methods=['POST'])
def receive_face():
    file_list = []
    # print(request.form)
    # if ('user_id' not in request.form):
    #     return jsonify({'error': 'no user id.'})
    # user_id = request.form['user_id']
    user_id = get_random_string()

    if ('face' not in request.files):
        return jsonify({'error': 'no face image file.'})
    face_image = request.files['face']

    if face_image.filename == '':
        return jsonify({'error': 'no face image file.'})

    print(face_image.filename)
    img_path = upload_request_images_face(face_image, user_id, 'liveness')

    res = check_live(img_path)

    if 'error' in res:
        return jsonify(res)

    answer = res['face_liveness']
    return jsonify(build_face_liveness_string_response(answer))


@app.route('/face_liveness_base64', methods=['POST'])
def receive_face_base64():
    data = request.get_json(silent=True)

    if not data or 'image_base64' not in data:
        return jsonify({'error': 'image_base64 is required'}), 400

    image_base64 = data['image_base64']

    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        return jsonify({'error': 'Invalid base64 image'}), 400

    try:
        suffix = get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    img_path = save_request_image_bytes(image_bytes, suffix)

    # Reuse existing logic
    res = check_live(img_path)

    if 'error' in res:
        return jsonify(res)

    answer = res['face_liveness']
    return jsonify(build_face_liveness_object_response(answer))


@app.route('/face_liveness_base64_batch', methods=['POST'])
def receive_face_base64_batch():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({'error': 'images_base64 is required'}), 400

    images_base64 = data.get('images_base64')
    if images_base64 is None:
        images_base64 = data.get('image_base64_list')

    if not isinstance(images_base64, list) or len(images_base64) == 0:
        return jsonify({'error': 'images_base64 must be a non-empty list'}), 400

    results = [None] * len(images_base64)
    valid_indices = []
    valid_paths = []

    for idx, image_base64 in enumerate(images_base64):
        if not isinstance(image_base64, str) or not image_base64.strip():
            results[idx] = {'success': False, 'error': 'image_base64 must be a non-empty string'}
            continue

        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            results[idx] = {'success': False, 'error': 'Invalid base64 image'}
            continue

        try:
            suffix = get_image_suffix_from_bytes(image_bytes)
        except ValueError as exc:
            results[idx] = {'success': False, 'error': str(exc)}
            continue

        img_path = save_request_image_bytes(image_bytes, suffix)
        valid_indices.append(idx)
        valid_paths.append(img_path)

    if valid_paths:
        try:
            batch_results = check_live_batch(valid_paths)
        except Exception as exc:
            batch_results = [{'error': str(exc)} for _ in valid_paths]

        for local_idx, original_idx in enumerate(valid_indices):
            result = batch_results[local_idx]
            if result is None:
                results[original_idx] = {'success': False, 'error': 'Unknown batch inference error'}
            elif 'error' in result:
                results[original_idx] = {'success': False, 'error': result['error']}
            else:
                answer = result['face_liveness']
                results[original_idx] = build_face_liveness_object_response(answer)

    for idx, result in enumerate(results):
        if result is None:
            results[idx] = {'success': False, 'error': 'Unknown input error'}

    return jsonify({
        'success': True,
        'results': results,
    })

def get_ssl_context():
    cert_path = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "certs/key.pem")

    if (
        os.path.isfile(cert_path)
        and os.path.isfile(key_path)
        and os.access(cert_path, os.R_OK)
        and os.access(key_path, os.R_OK)
    ):
        return (cert_path, key_path)

    print(f"SSL cert/key not readable ({cert_path}, {key_path}); using ad-hoc self-signed cert.")
    return "adhoc"


if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    # app.run(host='127.0.0.1', port=5000)
    app.run(host='0.0.0.0', port=3000, ssl_context=get_ssl_context())
