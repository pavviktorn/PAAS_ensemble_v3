import ntpath
import random
import string
import json
import time
from concurrent.futures import Future, TimeoutError
from queue import Queue, Empty
from threading import Thread
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import base64
import os
from PIL import Image, UnidentifiedImageError
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import torch
# torch.set_flush_denormal(True)
from models import *
from transformers import AutoTokenizer, CLIPProcessor

from fastapi import Body, FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from werkzeug.utils import secure_filename

# from insightface.app import FaceAnalysis
from yolo11_cls_onnx import Yolo11ClsONNX
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

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(4096 * 4096)))
PROMPT_PATH = os.environ.get("LIVENESS_PROMPT_PATH", "playground/prompts.txt")
DYNAMIC_BATCHING = os.environ.get("DYNAMIC_BATCHING", "1") == "1"
DYNAMIC_BATCH_MAX_WAIT_MS = int(os.environ.get("DYNAMIC_BATCH_MAX_WAIT_MS", "120"))
INFERENCE_TIMEOUT_SEC = int(os.environ.get("INFERENCE_TIMEOUT_SEC", "180"))
_prompt_cache = {"mtime": None, "prompt": DEFAULT_LIVENESS_PROMPT}

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
    'mistral': "checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_1",
    'phi': "checkpoints_phi3/effaa-llava-phi-3-mini-4b-lora_3",
}
# mids path
# mids_path = f"checkpoints/effaa-llava-mistral-7b-lora_6/mids.pth"
# mids_path = f"checkpoints_4+13fmt/effaa-llava-mistral-7b-lora_2/mids.pth"
# mids_path = f"checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_1/mids.pth"
mids_path = f"checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_1/mids.pth"

g_crop = 0
device_id = 0

# load mllm
mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["mistral"], device_id)
# mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(llava_groups["phi"], device_id)

# load mids
t5_tokenizer = AutoTokenizer.from_pretrained('models/t5-base', use_fast=False, legacy=False)
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14-336")
mids = load_mids(mids_path, device_id)
mistral_model.eval()
mids.eval()
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("ALLOW_TF32", "1") == "1"
    torch.backends.cudnn.allow_tf32 = os.environ.get("ALLOW_TF32", "1") == "1"


tags_metadata = [
    {
        "name": "status",
        "description": "Health, runtime configuration, and batching diagnostics.",
    },
    {
        "name": "liveness",
        "description": "Face liveness/personhood analysis endpoints. These preserve the original Flask response formats.",
    },
]

app = FastAPI(
    title="PAAS Face Liveness API",
    version=os.environ.get("API_VERSION", "3.2-fastapi-json"),
    description=(
        "PAAS face liveness API with dynamic GPU request batching and JSON-object face_liveness responses. "
        "Swagger UI is available at `/docs`; ReDoc is available at `/redoc`; "
        "the OpenAPI schema is available at `/openapi.json`."
    ),
    docs_url=os.environ.get("SWAGGER_DOCS_URL", "/docs"),
    redoc_url=os.environ.get("REDOC_URL", "/redoc"),
    openapi_url=os.environ.get("OPENAPI_URL", "/openapi.json"),
    openapi_tags=tags_metadata,
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "defaultModelsExpandDepth": 1,
        "defaultModelExpandDepth": 2,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class Base64ImageRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded image. Data URL prefixes such as data:image/jpeg;base64,... are accepted.",
        examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."],
    )


class Base64BatchRequest(BaseModel):
    images_base64: Optional[List[str]] = Field(
        default=None,
        description="List of base64-encoded images. Preferred field name.",
        examples=[["/9j/4AAQSkZJRgABAQAAAQABAAD...", "iVBORw0KGgoAAAANSUhEUg..."]],
    )
    image_base64_list: Optional[List[str]] = Field(
        default=None,
        description="Backward-compatible alias for images_base64.",
    )


class HealthResponse(BaseModel):
    success: bool
    status: str


class BatcherStatusResponse(BaseModel):
    dynamic_batching: bool
    max_batch_size: int
    max_wait_ms: int
    queue_size: int
    max_new_tokens: int
    inference_timeout_sec: int
    allow_tf32: bool


class GenericLivenessResponse(BaseModel):
    success: bool = True
    face_liveness: Dict[str, Any] = Field(
        ...,
        description="Parsed liveness result as a JSON object, not a JSON-encoded string.",
        examples=[{
            "Image description": {"focus": "focused", "resolution": "Good", "quality": "Good"},
            "Forgery reasoning": "Natural features, consistent lighting, no manipulation.",
            "Analysis result": "real",
            "Forgery type": "None",
            "Match score": "1.0000",
            "Difficulty": "easy"
        }],
    )


class BatchLivenessResponse(BaseModel):
    success: bool = True
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Per-image results. Successful items contain face_liveness as a JSON object; failed items contain success=false and error.",
    )

@app.middleware("http")
async def set_response_headers(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Process-Time-Ms"] = f"{(time.time() - started) * 1000:.2f}"
    return response


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    return (
        "<html><body><h3>PAAS Face Liveness API</h3>"
        "<p>FastAPI server is running.</p>"
        "<ul>"
        "<li><a href='/docs'>Swagger UI</a></li>"
        "<li><a href='/redoc'>ReDoc</a></li>"
        "<li><a href='/batcher_status'>Batcher status</a></li>"
        "</ul></body></html>"
    )


@app.get("/health", response_model=HealthResponse, tags=["status"], summary="Health check")
async def health():
    return {"success": True, "status": "ok"}


@app.get("/batcher_status", response_model=BatcherStatusResponse, tags=["status"], summary="Show batching/runtime settings")
async def batcher_status():
    return {
        "dynamic_batching": inference_batcher is not None,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_wait_ms": DYNAMIC_BATCH_MAX_WAIT_MS,
        "queue_size": inference_batcher.queue.qsize() if inference_batcher is not None else 0,
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "64")),
        "inference_timeout_sec": INFERENCE_TIMEOUT_SEC,
        "allow_tf32": os.environ.get("ALLOW_TF32", "1") == "1",
    }


def getRotatedAngle(img):
    if img is None:
        raise ValueError("Failed to read image")

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


def get_request_image_subdir(request_obj: Request | None = None):
    if request_obj is not None:
        ipaddr = request_obj.headers.get("X-Forwarded-For") or (request_obj.client.host if request_obj.client else None)
    else:
        ipaddr = "unknown"
    if ipaddr:
        ipaddr = ipaddr.split(",")[0].strip()
    if not ipaddr:
        ipaddr = "unknown"

    subdir = os.path.join('./images', ipaddr)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def get_image_suffix_from_bytes(image_bytes):
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            # if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            #     raise ValueError("Image dimensions are not supported")
            image.load()
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Failed to decode image") from exc

    return IMAGE_FORMAT_TO_EXT.get(image_format, ".png")


def save_request_image_bytes(image_bytes, suffix, request_obj: Request | None = None):
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix
    file_path = os.path.join(get_request_image_subdir(request_obj), fname)
    with open(file_path, "wb") as file_obj:
        file_obj.write(image_bytes)
    return file_path


def get_random_string():
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(8))
    return result_str


def get_liveness_prompt():
    """Read the prompt only when the file changes."""
    try:
        mtime = os.path.getmtime(PROMPT_PATH)
    except OSError:
        return DEFAULT_LIVENESS_PROMPT

    if _prompt_cache["mtime"] != mtime:
        prompt_list = read_txt_file(PROMPT_PATH)
        _prompt_cache["prompt"] = prompt_list[0] if prompt_list else DEFAULT_LIVENESS_PROMPT
        _prompt_cache["mtime"] = mtime

    return _prompt_cache["prompt"]


def build_face_liveness_response(answer):
    """Return face_liveness as a real JSON object, not a JSON-encoded string."""
    output = get_jsonfmt(answer)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return {
        'success': True,
        'face_liveness': output,
    }


# Backward-compatible aliases for older internal call sites.
def build_face_liveness_string_response(answer):
    return build_face_liveness_response(answer)


def build_face_liveness_object_response(answer):
    return build_face_liveness_response(answer)


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
    request_batch_size = len(image_paths)
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
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "64")),
        "generate_num": genarate_num_dict,
    })()

    start_t = time.time()
    with torch.inference_mode():
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
    print(f"batch total time : {end_t - start_t}s, MIDS : {end_t - mids_s}, request_batch_size : {request_batch_size}, valid_batch_size : {len(mids_images)}")

    return results



class DynamicInferenceBatcher:
    """Coalesces concurrent HTTP requests into one GPU batch.

    This improves throughput under parallel traffic. A single request may not become
    faster, because the model still needs the same three generation passes.
    """

    def __init__(self, max_batch_size=8, max_wait_ms=35, timeout_sec=180):
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms) / 1000.0)
        self.timeout_sec = timeout_sec
        self.queue = Queue()
        self.worker = Thread(target=self._worker_loop, daemon=True, name="dynamic-inference-batcher")
        self.worker.start()

    def submit(self, image_path):
        return self.submit_many([image_path])[0]

    def submit_many(self, image_paths):
        future = Future()
        self.queue.put((list(image_paths), future))
        return future.result(timeout=self.timeout_sec)

    def _worker_loop(self):
        while True:
            first_paths, first_future = self.queue.get()
            batch_items = [(first_paths, first_future)]
            total = len(first_paths)
            deadline = time.time() + self.max_wait_s

            while total < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    paths, future = self.queue.get(timeout=remaining)
                except Empty:
                    break
                batch_items.append((paths, future))
                total += len(paths)

            # Drain any requests that arrived just after the wait window without
            # adding more latency. This helps when HTTP upload/preprocessing jitter
            # spreads parallel requests by a few milliseconds.
            while total < self.max_batch_size:
                try:
                    paths, future = self.queue.get_nowait()
                except Empty:
                    break
                batch_items.append((paths, future))
                total += len(paths)

            flat_paths = []
            slices = []
            cursor = 0
            for paths, _future in batch_items:
                flat_paths.extend(paths)
                slices.append((cursor, cursor + len(paths)))
                cursor += len(paths)

            print(f"dynamic batcher dispatching: grouped_requests={len(batch_items)}, images={len(flat_paths)}, max_wait_ms={int(self.max_wait_s * 1000)}")
            try:
                batch_results = check_live_batch(flat_paths)
            except Exception as exc:
                for _paths, future in batch_items:
                    if not future.done():
                        future.set_exception(exc)
                continue

            for (_paths, future), (start, end) in zip(batch_items, slices):
                if not future.done():
                    future.set_result(batch_results[start:end])


inference_batcher = DynamicInferenceBatcher(
    max_batch_size=MAX_BATCH_SIZE,
    max_wait_ms=DYNAMIC_BATCH_MAX_WAIT_MS,
    timeout_sec=INFERENCE_TIMEOUT_SEC,
) if DYNAMIC_BATCHING else None


def check_live_dynamic(image_path):
    if inference_batcher is None:
        return check_live(image_path)
    return inference_batcher.submit(image_path)


def check_live_batch_dynamic(image_paths):
    if inference_batcher is None:
        return check_live_batch(image_paths)
    return inference_batcher.submit_many(image_paths)

def inference(args):
    # args
    image_path = args.image_path
    crop = args.crop

    mids.eval()

    start_t = time.time()

    print(f'USER: {args.prompt}\n')
    device = torch.device(f'cuda:{device_id}')

    def run_mistral(image, answer_holder):
        with torch.inference_mode():
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
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "64"))
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


@app.post('/face_liveness', response_model=GenericLivenessResponse, tags=['liveness'], summary='Analyze uploaded face image')
async def receive_face(request: Request, face: UploadFile = File(..., description='Face image file')):
    if not face.filename:
        raise HTTPException(status_code=400, detail='no face image file.')

    print(face.filename)
    file_name = secure_filename(face.filename)
    _name, ext = os.path.splitext(file_name)
    if not ext:
        ext = '.jpg'
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
    subdir = get_request_image_subdir(request)
    file_path = os.path.join(subdir, fname)

    image_bytes = await face.read()
    try:
        get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    with open(file_path, 'wb') as file_obj:
        file_obj.write(image_bytes)

    try:
        res = await run_in_threadpool(check_live_dynamic, file_path)
    except TimeoutError:
        raise HTTPException(status_code=504, detail='Inference timed out')

    if 'error' in res:
        return JSONResponse(res)

    answer = res['face_liveness']
    return JSONResponse(build_face_liveness_response(answer))


@app.post('/face_liveness_base64', response_model=GenericLivenessResponse, tags=['liveness'], summary='Analyze base64 face image')
async def receive_face_base64(request: Request, payload: Base64ImageRequest = Body(...)):
    data = payload.model_dump()

    if not data or 'image_base64' not in data:
        raise HTTPException(status_code=400, detail='image_base64 is required')

    image_base64 = data['image_base64']
    if ',' in image_base64:
        image_base64 = image_base64.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid base64 image')

    try:
        suffix = get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    img_path = save_request_image_bytes(image_bytes, suffix, request)

    try:
        res = await run_in_threadpool(check_live_dynamic, img_path)
    except TimeoutError:
        raise HTTPException(status_code=504, detail='Inference timed out')

    if 'error' in res:
        return JSONResponse(res)

    answer = res['face_liveness']
    return JSONResponse(build_face_liveness_response(answer))


@app.post('/face_liveness_base64_batch', response_model=BatchLivenessResponse, tags=['liveness'], summary='Analyze a batch of base64 face images')
async def receive_face_base64_batch(request: Request, payload: Base64BatchRequest = Body(...)):
    data = payload.model_dump()

    if not data:
        raise HTTPException(status_code=400, detail='images_base64 is required')

    images_base64 = data.get('images_base64')
    if images_base64 is None:
        images_base64 = data.get('image_base64_list')

    if not isinstance(images_base64, list) or len(images_base64) == 0:
        raise HTTPException(status_code=400, detail='images_base64 must be a non-empty list')
    if len(images_base64) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f'Batch size exceeds limit of {MAX_BATCH_SIZE}')

    results = [None] * len(images_base64)
    valid_indices = []
    valid_paths = []

    for idx, image_base64 in enumerate(images_base64):
        if not isinstance(image_base64, str) or not image_base64.strip():
            results[idx] = {'success': False, 'error': 'image_base64 must be a non-empty string'}
            continue

        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]

        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            results[idx] = {'success': False, 'error': 'Invalid base64 image'}
            continue

        try:
            suffix = get_image_suffix_from_bytes(image_bytes)
        except ValueError as exc:
            results[idx] = {'success': False, 'error': str(exc)}
            continue

        img_path = save_request_image_bytes(image_bytes, suffix, request)
        valid_indices.append(idx)
        valid_paths.append(img_path)

    if valid_paths:
        try:
            batch_results = await run_in_threadpool(check_live_batch_dynamic, valid_paths)
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
                results[original_idx] = build_face_liveness_response(answer)

    for idx, result in enumerate(results):
        if result is None:
            results[idx] = {'success': False, 'error': 'Unknown input error'}

    return JSONResponse({'success': True, 'results': results})


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
    import uvicorn

    cert_path = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "certs/key.pem")
    ssl_kwargs = {}
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        ssl_kwargs = {"ssl_certfile": cert_path, "ssl_keyfile": key_path}

    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "3000")),
        workers=1,
        **ssl_kwargs,
    )
