#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoTokenizer, CLIPProcessor

transformers.logging.set_verbosity_error()

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from mids.selector import make_decision_batch
from utils.file_utils import decode_response, get_jsonfmt, mask_result, read_txt_file


DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/axonlabs_data_1"
DEFAULT_MISS_DIR = "/datasets/work/vLLM/data/miss_axonlabs_data_1"
DEFAULT_MODEL_PATH = "checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_1"
DEFAULT_PROMPT = "The image is a human face image. Is it real or fake? Why?"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
LABELS = ("real", "fake")
JPEG_QUALITY = 95

cv2: Any = None
np: Any = None


# ------------------------------------------------------------- real face-quality filter
class FaceQualityFilter:
    """Keep only frames with ONE dominant, frontal, proper-size, unoccluded face.
    insightface buffalo_l (SCRFD detection + 3D-68 landmark/pose) on CPU. Used to skip
    low-quality REAL images/frames (heavy head pose / wrong face size / occluded); never
    applied to fakes. Mirrors the MIDS++ ensemble tool's filter."""

    def __init__(self, score, pose, hmin, hmax, minpx, maxside, det_size=640):
        from insightface.app import FaceAnalysis
        self.score, self.pose = float(score), float(pose)
        self.hmin, self.hmax = float(hmin), float(hmax)
        self.minpx, self.maxside = int(minpx), int(maxside)
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def _downscale(self, bgr):
        import cv2 as _cv2
        h, w = bgr.shape[:2]
        sc = self.maxside / max(h, w)
        return _cv2.resize(bgr, (int(w * sc), int(h * sc)),
                           interpolation=_cv2.INTER_AREA) if sc < 1 else bgr

    def passes(self, bgr):
        """Return (keep: bool, reason: str). reason is the first failing check (or 'ok')."""
        import numpy as _np
        img = self._downscale(bgr)
        h = img.shape[0]
        faces = self.app.get(img)
        if not faces:
            return False, "noface"
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        f = faces[0]
        x1, y1, x2, y2 = f.bbox
        fw, fh = x2 - x1, y2 - y1
        area = fw * fh
        if len(faces) > 1:
            f2 = faces[1]
            if (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1]) > 0.5 * area:
                return False, "multiface"
        if f.det_score < self.score:
            return False, "lowdet"
        if not (self.hmin <= fh / h <= self.hmax):
            return False, "size"
        if min(fw, fh) < self.minpx:
            return False, "minpx"
        pose = getattr(f, "pose", None)
        if pose is None or float(_np.max(_np.abs(pose))) > self.pose:
            return False, "pose"
        return True, "ok"


def subset_for(source_path: Path) -> str:
    """Per-subset key for the summary: the folder right under the real/fake label dir, else the label."""
    i = get_label_index(source_path)
    if i is not None and i + 1 < len(source_path.parts) - 1:
        return source_path.parts[i + 1]
    return get_true_label_from_path(source_path) or "unknown"


def results_line(tag: str, true_label: str, pred: Any, ftype: Any,
                 fake_score: Any, match_score: Any, path: str) -> str:
    """One consolidated-results line in the MIDS++ ensemble column layout (results_1.txt format):
        tag  truth=..  pred=..  type=..  fake=..  match=..  path
    fake_score/match_score are floats for scored lines, or the string '------' for SK/ER lines."""
    ff = f"{fake_score:.4f}" if isinstance(fake_score, (int, float)) else str(fake_score)
    ms = f"{match_score:.4f}" if isinstance(match_score, (int, float)) else str(match_score)
    return (f"{tag}  truth={true_label}  pred={pred}  type={str(ftype):<8}  "
            f"fake={ff} match={ms}  {path}\n")


def merge_subtally(target: dict, source: dict) -> None:
    for subset, st in source.items():
        dst = target.setdefault(
            subset, {"group": st.get("group", "unknown"), "n": 0, "correct": 0, "skipped": 0})
        if st.get("group"):
            dst["group"] = st["group"]
        dst["n"] += int(st.get("n", 0))
        dst["correct"] += int(st.get("correct", 0))
        dst["skipped"] += int(st.get("skipped", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process every image and video frame under a folder with local "
            "LLaVA/MIDS checkpoints, using one worker per GPU by default. "
            "Misclassified images/frames are copied to the miss folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to scan for image/video files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--miss-dir",
        "--miss_dir",
        dest="miss_dir",
        default=DEFAULT_MISS_DIR,
        help=f"Folder used to save missed images. Default: {DEFAULT_MISS_DIR}",
    )
    parser.add_argument(
        "--devices",
        default="all",
        help="CUDA devices to use, e.g. 'all' or '0,1,2,3'. Default: all",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Single-GPU alias. If set, overrides --devices.",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the LLaVA checkpoint. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--mids-path",
        "--mids_path",
        dest="mids_path",
        default=None,
        help="Path to MIDS checkpoint. Default: <model-path>/mids.pth",
    )
    parser.add_argument(
        "--prompt-list",
        "--prompt_list",
        dest="prompt_list",
        default="playground/prompts.txt",
        help="Path to prompts. The first prompt is used, matching the API path.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        "--workers",
        dest="batch_size",
        type=int,
        default=160,
        help="Images/frames per GPU batch. Default: 160",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help=(
            "Approx max queued image/frame capacity. Internally queued in batches "
            "to reduce GPU-worker dequeue idle time. 0 chooses batch_size * gpu_count * 16."
        ),
    )
    parser.add_argument(
        "--producer-workers",
        type=int,
        default=32,
        help=(
            "Number of CPU producer processes for image/video decode. "
            "0 auto-selects up to the GPU count. Default: 0"
        ),
    )
    parser.add_argument(
        "--queue-report-interval",
        type=float,
        default=10.0,
        help=(
            "Seconds between work-queue fullness reports. Use 0 to disable. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--print-ok",
        action="store_true",
        help="Print every correctly classified item. Disabled by default for speed.",
    )
    parser.add_argument(
        "--skip-existing",
        type=int,
        default=1,
        help=(
            "if 1, Skip still images with an existing .txt output and skip videos with "
            "an existing .frames.jsonl output. if 0, no skipping."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=None)
    parser.add_argument("--num-beams", "--num_beams", dest="num_beams", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--mistral-generations",
        "--mistral_generations",
        dest="mistral_generations",
        type=int,
        default=3,
        help="Number of conditional LLaVA answers per image. Default: 3",
    )
    parser.add_argument(
        "--t5-model-path",
        "--t5_model_path",
        dest="t5_model_path",
        default="models/t5-base",
        help="Path to the T5 text encoder used by MIDS.",
    )
    parser.add_argument(
        "--clip-model-path",
        "--clip_model_path",
        dest="clip_model_path",
        default="models/clip-vit-large-patch14-336",
        help="Path to the CLIP image encoder used by MIDS.",
    )
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"),
        help="Consolidated per-item results + summary file (OK/XX/SK/ER lines). "
             "Default: results.txt next to this script.",
    )
    # --- real face-quality filter (heavy pose / wrong face size / occluded), images & frames ---
    parser.add_argument(
        "--filter-real", "--filter_real", dest="filter_real", type=int, default=1,
        help="1: skip low-quality REAL images/frames (non-frontal / wrong face size / occluded / "
             "no single dominant face) and EXCLUDE them from accuracy. 0: keep all. "
             "PAD/fake items are never filtered.",
    )
    parser.add_argument("--face-score", dest="face_score", type=float, default=0.65,
                        help="Min detector confidence (occlusion/quality proxy).")
    parser.add_argument("--face-pose", dest="face_pose", type=float, default=28.0,
                        help="Max |pitch|,|yaw|,|roll| degrees (frontal).")
    parser.add_argument("--face-hmin", dest="face_hmin", type=float, default=0.15,
                        help="Min face-height fraction of the frame.")
    parser.add_argument("--face-hmax", dest="face_hmax", type=float, default=0.85,
                        help="Max face-height fraction of the frame.")
    parser.add_argument("--face-minpx", dest="face_minpx", type=int, default=80,
                        help="Min face side in pixels (on the downscaled frame).")
    parser.add_argument("--face-maxside", dest="face_maxside", type=int, default=1280,
                        help="Downscale the long side to this before face detection.")
    return parser.parse_args()


def parse_devices(args: argparse.Namespace) -> list[int]:
    if args.device is not None:
        return [int(args.device)]

    visible_count = torch.cuda.device_count()
    if args.devices.strip().lower() == "all":
        return list(range(visible_count))

    devices = []
    for part in args.devices.split(","):
        part = part.strip()
        if not part:
            continue
        devices.append(int(part))

    return devices


def load_image_dependencies() -> bool:
    global cv2
    global np

    if cv2 is not None and np is not None:
        return True

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ModuleNotFoundError as exc:
        print(
            f"Missing required Python package: {exc.name}. "
            "Activate the environment with opencv-python and numpy installed.",
            file=sys.stderr,
        )
        return False

    cv2 = cv2_module
    np = np_module
    return True


def build_label_stats() -> dict[str, dict[str, int]]:
    return {
        label: {
            "found": 0,
            "processed": 0,
            "evaluated": 0,
            "correct": 0,
            "miss": 0,
            "failed": 0,
            "skipped_quality": 0,
        }
        for label in LABELS
    }


def build_counts() -> dict[str, int]:
    return {
        "total_items": 0,
        "still_images_processed": 0,
        "video_frames_extracted": 0,
        "correct": 0,
        "incorrect": 0,
        "failed": 0,
        "miss_saved": 0,
        "failed_sources": 0,
        "skipped_sources": 0,
        "real_skipped_quality": 0,
        "total_batches": 0,
        "worker_errors": 0,
        "producer_errors": 0,
        "writer_errors": 0,
    }


def merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def merge_label_stats(
    target: dict[str, dict[str, int]],
    source: dict[str, dict[str, int]],
) -> None:
    for label, stats in source.items():
        if label not in target:
            target[label] = {}
        for key, value in stats.items():
            target[label][key] = target[label].get(key, 0) + int(value)


def iter_media_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
            yield path


def get_label_index(path: Path) -> Optional[int]:
    for index, part in enumerate(path.parts[:-1]):
        lower_part = part.lower()
        if lower_part == "real" or lower_part == "fake":
            return index
    return None


def get_true_label_from_path(path: Path) -> Optional[str]:
    label_index = get_label_index(path)
    if label_index is None:
        return None
    return path.parts[label_index].lower()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def get_source_output_path(source_path: Path) -> Path:
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return source_path.with_suffix(".txt")
    return Path(f"{source_path}.frames.jsonl")


def has_nonempty_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def dump_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False))
        handle.write("\n")


def response_to_text(response_json: Any) -> str:
    return json.dumps(response_json, indent=2, ensure_ascii=False)


def parse_json_text(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_analysis_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in {"analysis_result", "analysisresult", "result"}:
            return value.strip().strip('"').strip("'")
    return None


def extract_analysis_result(response_json: Any) -> Optional[str]:
    candidates = []
    if isinstance(response_json, dict):
        candidates.append(response_json)
        face_liveness = response_json.get("face_liveness")
        if face_liveness is not None:
            candidates.append(face_liveness)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("Analysis result", "analysis_result", "analysisresult", "result"):
                if key in candidate:
                    value = candidate[key]
                    return str(value).strip() if value is not None else None
        elif isinstance(candidate, str):
            nested_json = parse_json_text(candidate)
            if nested_json is not None:
                nested_result = extract_analysis_result(nested_json)
                if nested_result is not None:
                    return nested_result
            text_result = extract_analysis_from_text(candidate)
            if text_result is not None:
                return text_result

    if isinstance(response_json, str):
        return extract_analysis_from_text(response_json)

    return None


def extract_match_score(response_json: Any) -> Optional[str]:
    candidates = []
    if isinstance(response_json, dict):
        candidates.append(response_json)
        face_liveness = response_json.get("face_liveness")
        if face_liveness is not None:
            candidates.append(face_liveness)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("Match score", "match_score", "matchscore"):
                if key in candidate:
                    value = candidate[key]
                    return str(value).strip() if value is not None else None
        elif isinstance(candidate, str):
            nested_json = parse_json_text(candidate)
            if nested_json is not None:
                nested_score = extract_match_score(nested_json)
                if nested_score is not None:
                    return nested_score
            for line in candidate.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                normalized_key = key.strip().lower().replace(" ", "_")
                if normalized_key in {"match_score", "matchscore"}:
                    return value.strip().strip('"').strip("'")

    if isinstance(response_json, str):
        for line in response_json.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {"match_score", "matchscore"}:
                return value.strip().strip('"').strip("'")

    return None


def normalize_prediction_label(analysis_result: Optional[str]) -> str:
    if analysis_result is None:
        return "fake"

    normalized = analysis_result.strip().lower()
    if normalized == "real":
        return "real"
    if normalized in ("ambiguous", "likely_fake"):
        return "ambiguous"
    return "fake"


def relative_parent_for_source(source_path: Path, input_dir: Path) -> Path:
    try:
        return source_path.relative_to(input_dir).parent
    except ValueError:
        label_index = get_label_index(source_path)
        if label_index is not None:
            return Path(*source_path.parts[label_index:-1])
        return source_path.parent


def infer_source_name(source_path: Path, input_dir: Path) -> str:
    try:
        parent_parts = source_path.relative_to(input_dir).parent.parts
    except ValueError:
        parent_parts = source_path.parent.parts

    for part in reversed(parent_parts):
        if part.lower() not in LABELS:
            return part
    return source_path.stem


def unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        next_candidate = candidate.with_name(f"{stem}_{counter}{suffix}")
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def encode_image_bytes_as_jpeg(image_bytes: bytes) -> bytes:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode image")

    success, jpeg = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not success:
        raise RuntimeError("failed to encode image as JPEG")
    return jpeg.tobytes()


def build_miss_output_path(item: dict[str, Any], miss_dir: Path) -> Path:
    candidate = miss_dir / Path(item["relative_parent"]) / item["miss_filename"]
    return unique_path(candidate)


def save_missed_image(item: dict[str, Any], miss_dir: Path, payload_bytes: bytes) -> Path:
    output_path = build_miss_output_path(item, miss_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if item["kind"] == "video_frame":
        miss_bytes = payload_bytes
    else:
        miss_bytes = encode_image_bytes_as_jpeg(payload_bytes)

    output_path.write_bytes(miss_bytes)
    return output_path


def build_video_frame_record(
    item: dict[str, Any],
    result: Any,
    analysis_result: Optional[str],
    predicted_label: str,
    match_score: Optional[str],
    miss_output_path: Optional[Path],
) -> dict[str, Any]:
    return {
        "source_video": item["source_path"],
        "frame_index": item["frame_index"],
        "frame_name": item["miss_filename"],
        "true_label": item["true_label"],
        "analysis_result": analysis_result,
        "predicted_label": predicted_label,
        "match_score": match_score,
        "miss_copy": str(miss_output_path) if miss_output_path is not None else None,
        "response": result,
    }


def save_result_for_item(
    item: dict[str, Any],
    result: Any,
    analysis_result: Optional[str],
    predicted_label: str,
    match_score: Optional[str],
    miss_output_path: Optional[Path],
) -> None:
    output_path = Path(item["output_path"])
    if item["kind"] == "image":
        if isinstance(result, dict):
            dump_text_file(output_path, response_to_text(result))
        else:
            dump_text_file(output_path, json.dumps(result, ensure_ascii=False, indent=2))
        return

    append_jsonl(
        output_path,
        build_video_frame_record(
            item,
            result,
            analysis_result,
            predicted_label,
            match_score,
            miss_output_path,
        ),
    )


def save_failure_result_for_item(item: dict[str, Any], message: str) -> None:
    error_json = {
        "success": False,
        "error": message,
        "source": item["source_path"],
    }

    if item["kind"] == "image":
        dump_text_file(Path(item["output_path"]), response_to_text(error_json))
        return

    record = build_video_frame_record(
        item,
        error_json,
        None,
        "fake",
        None,
        None,
    )
    append_jsonl(Path(item["output_path"]), record)


def save_quality_skip_for_item(item: dict[str, Any], reason: str) -> None:
    """Record a low-quality REAL image/frame skip (excluded from accuracy)."""
    record = {
        "success": False,
        "skipped": "low_quality_real",
        "reason": reason,
        "source": item["source_path"],
    }
    if item["kind"] == "image":
        dump_text_file(Path(item["output_path"]), response_to_text(record))
        return
    record["frame_index"] = item.get("frame_index")
    append_jsonl(Path(item["output_path"]), record)


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model, image_processor, tokenizer


def load_mids(mids_path: str, device_id: int, t5_model_path: str, clip_model_path: str):
    from mids.mids_arch import MIDS

    model = MIDS(text_model_path=t5_model_path, image_model_path=clip_model_path)
    model_state_dict = model.state_dict()
    finetuned_state_dict = torch.load(mids_path, map_location="cpu")
    finetuned_state_dict = {
        key.replace("module.", ""): value for key, value in finetuned_state_dict.items()
    }
    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)
    return model.to(dtype=torch.float32, device=torch.device(f"cuda:{device_id}"))


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
def generate_batch_with_conditionals(
    model,
    tokenizer,
    image_processor,
    images: list[Any],
    prompts: list[str],
    conv_mode: str = "v1",
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    per_sample_generate_num: int = 3,
) -> list[list[str]]:
    if len(images) != len(prompts):
        raise ValueError("images and prompts must align")
    if not images:
        return []

    image_sizes = [image.size for image in images]
    image_tensor = process_images(
        images,
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)

    def build_input_ids(qs_list: list[str]) -> torch.Tensor:
        ids = []
        for qs in qs_list:
            full_prompt = get_llava_prompt(model, qs, conv_mode)
            tokens = tokenizer_image_token(
                full_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            )
            if tokens.ndim > 1:
                tokens = tokens.squeeze(0)
            ids.append(tokens)
        return torch.nn.utils.rnn.pad_sequence(
            ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(model.device)

    def run_generation(qs_list: list[str]) -> list[str]:
        input_ids = build_input_ids(qs_list)
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
        return [
            text.strip()
            for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]

    outputs = [[] for _ in images]
    decoded_1 = run_generation(prompts)
    for index, text in enumerate(decoded_1):
        outputs[index].append(text)

    if per_sample_generate_num <= 1:
        return outputs

    condition_prompt = "This is a _ human face. What evidence do you have?"
    prompts_2 = []
    prompts_3 = []
    for answers in outputs:
        try:
            response_json, _ = decode_response(answers[0])
            answer_result = response_json["Analysis result"].lower()
        except Exception:
            answer_result = "fake"

        if answer_result == "real":
            prompts_2.append(condition_prompt.replace("_", "fake"))
            prompts_3.append(condition_prompt.replace("_", "real"))
        else:
            prompts_2.append(condition_prompt.replace("_", "real"))
            prompts_3.append(condition_prompt.replace("_", "fake"))

    decoded_2 = run_generation(prompts_2)
    for index, text in enumerate(decoded_2):
        outputs[index].append(text)

    if per_sample_generate_num <= 2:
        return outputs

    decoded_3 = run_generation(prompts_3)
    for index, text in enumerate(decoded_3):
        outputs[index].append(text)

    return [answers[:per_sample_generate_num] for answers in outputs]


def load_prompt(prompt_list_path: str) -> str:
    try:
        prompts = read_txt_file(prompt_list_path)
    except FileNotFoundError:
        prompts = []
    return prompts[0] if prompts else DEFAULT_PROMPT


def image_from_item(item: dict[str, Any]) -> Image.Image:
    return Image.open(BytesIO(item["payload_bytes"])).convert("RGB")


def format_answer_json(answer_json: dict[str, Any]) -> str:
    data = dict(answer_json)
    data.pop("Probability", None)
    key_order = [
        "Image description",
        "Forgery reasoning",
        "Analysis result",
        "Forgery type",
        "Match score",
        "Difficulty",
    ]
    return "\n".join(f"{key}: {data[key]}" for key in key_order if key in data)


def parse_answer_text(answer: str) -> dict[str, Any]:
    try:
        parsed = get_jsonfmt(answer)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    parsed: dict[str, Any] = {}
    for line in answer.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if re.fullmatch(r"\d+\.\d+", value):
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed


def finalize_best_answer(
    answers: list[str],
    answers_result: list[str],
    best_answer_idx: int,
    match_score: float,
) -> tuple[str, dict[str, Any]]:
    if len(set(answers_result)) == 1:
        qs_difficulty = "easy"
    else:
        qs_difficulty = "hard"

    best_answer_json, _ = decode_response(answers[best_answer_idx])
    probability = float(best_answer_json.get("Probability", 1.0))

    if best_answer_json["Analysis result"].lower() == "real":
        if probability < 0.8 or (match_score < 0.99 and match_score > 0.8):
            if qs_difficulty == "hard":
                idx = len(answers_result) - 1
                for answer_result in reversed(answers_result):
                    if answer_result == "fake":
                        break
                    idx -= 1
                selected_json, _ = decode_response(answers[idx])
                best_answer_json["Forgery type"] = selected_json["Forgery type"]
                best_answer_json["Analysis result"] = "ambiguous"
                best_answer_json["Forgery reasoning"] = (
                    selected_json["Forgery reasoning"]
                    + ' It\'s close to "real", but not completely certain.'
                )
            else:
                best_answer_json["Analysis result"] = "ambiguous"
                best_answer_json["Forgery reasoning"] = (
                    best_answer_json["Forgery reasoning"]
                    + ' It\'s close to "real", but not completely certain.'
                )
        elif match_score <= 0.8:
            best_answer_json["Forgery type"] = "ambiguous"
            best_answer_json["Analysis result"] = "likely_fake"
            if qs_difficulty == "hard":
                idx = len(answers_result) - 1
                for answer_result in reversed(answers_result):
                    if answer_result == "fake":
                        break
                    idx -= 1
                selected_json, _ = decode_response(answers[idx])
                best_answer_json["Forgery type"] = selected_json["Forgery type"]
                best_answer_json["Forgery reasoning"] = (
                    selected_json["Forgery reasoning"]
                    + ' It\'s closer to "fake", but not completely certain.'
                )
            else:
                best_answer_json["Forgery reasoning"] = (
                    best_answer_json["Forgery reasoning"]
                    + ' It\'s closer to "fake", but not completely certain.'
                )

    best_answer_json["Match score"] = f"{match_score:.4f}"
    best_answer_json["Difficulty"] = qs_difficulty
    best_answer = format_answer_json(best_answer_json)
    return best_answer, best_answer_json


def mids_shape_for_generations(generation_count: int) -> tuple[int, int]:
    if generation_count >= 3:
        return 1, 1
    if generation_count == 2:
        return 0, 1
    return 0, 0


class LocalLivenessBatcher:
    def __init__(self, args: argparse.Namespace, device_id: int, worker_id: int) -> None:
        self.args = args
        self.device_id = int(device_id)
        self.worker_id = int(worker_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.prompt = load_prompt(args.prompt_list)

        torch.cuda.set_device(self.device_id)

        model_path = str(Path(args.model_path).expanduser())
        mids_path = args.mids_path or os.path.join(model_path, "mids.pth")
        self.model_path = model_path
        self.mids_path = str(Path(mids_path).expanduser())

        print(f"[GPU {self.device_id}] Loading LLaVA checkpoint: {self.model_path}", flush=True)
        self.model, self.image_processor, self.tokenizer = load_llava(
            self.model_path,
            self.device_id,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[GPU {self.device_id}] Loading MIDS checkpoint: {self.mids_path}", flush=True)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            args.t5_model_path,
            use_fast=False,
            legacy=False,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(args.clip_model_path)
        self.mids_model = load_mids(
            self.mids_path,
            self.device_id,
            args.t5_model_path,
            args.clip_model_path,
        )
        self.mids_model.eval()

    def infer(self, batch: list[dict[str, Any]]) -> list[Any]:
        results: list[Any] = [None] * len(batch)
        valid_indices: list[int] = []
        valid_images: list[Image.Image] = []

        for index, item in enumerate(batch):
            try:
                image = image_from_item(item)
            except Exception as exc:
                results[index] = {"success": False, "error": f"failed to decode image: {exc}"}
                continue

            valid_indices.append(index)
            valid_images.append(image)

        if not valid_images:
            return results

        answers_per_image = generate_batch_with_conditionals(
            model=self.model,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            images=valid_images,
            prompts=[self.prompt] * len(valid_images),
            conv_mode="v1",
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_new_tokens,
            per_sample_generate_num=self.args.mistral_generations,
        )

        mids_indices: list[int] = []
        mids_images: list[Image.Image] = []
        mids_answers: list[list[str]] = []
        answers_result_per_image: list[list[str]] = []
        flattened_answers_result: list[str] = []
        flattened_processed_answers: list[str] = []

        expected_count = max(1, self.args.mistral_generations)
        for local_index, image, answers in zip(valid_indices, valid_images, answers_per_image):
            if len(answers) != expected_count:
                results[local_index] = {
                    "success": False,
                    "error": f"expected {expected_count} generated answers, got {len(answers)}",
                }
                continue

            answers_result: list[str] = []
            processed_answers: list[str] = []
            try:
                for answer in answers:
                    masked_answer, answer_result = mask_result(answer)
                    answers_result.append(answer_result)
                    processed_answers.append(masked_answer)
            except Exception as exc:
                results[local_index] = {"success": False, "error": f"bad model output: {exc}"}
                continue

            mids_indices.append(local_index)
            mids_images.append(image)
            mids_answers.append(answers)
            answers_result_per_image.append(answers_result)
            flattened_answers_result.extend(answers_result)
            flattened_processed_answers.extend(processed_answers)

        if not mids_images:
            return results

        n_condition, m_condition = mids_shape_for_generations(expected_count)
        with torch.inference_mode():
            input_images = self.clip_processor(
                images=mids_images,
                return_tensors="pt",
            )["pixel_values"]
            answer_ids = self.t5_tokenizer(
                flattened_processed_answers,
                return_tensors="pt",
                padding="longest",
                max_length=self.t5_tokenizer.model_max_length,
                truncation=True,
            )
            logits = self.mids_model(
                answer_ids.to(self.device),
                input_images.to(self.device),
                None,
                len(mids_images),
                n_condition,
                m_condition,
            )["logits"]
            scores = F.softmax(logits, dim=2)
            best_answer_idxs, _preds, match_scores, _forgery_scores = make_decision_batch(
                flattened_answers_result,
                scores,
                chunk_size=expected_count,
            )

        for index, local_index in enumerate(mids_indices):
            try:
                best_answer, _best_answer_json = finalize_best_answer(
                    mids_answers[index],
                    answers_result_per_image[index],
                    int(best_answer_idxs[index]),
                    float(match_scores[index]),
                )
                results[local_index] = {
                    "success": True,
                    "face_liveness": parse_answer_text(best_answer),
                }
            except Exception as exc:
                results[local_index] = {"success": False, "error": f"failed to finalize answer: {exc}"}

        return results


def stripped_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "payload_bytes"}


def format_item_id(item: dict[str, Any]) -> str:
    producer_id = item.get("producer_id", "?")
    item_index = item.get("index", "?")
    return f"p{producer_id}:{item_index}"


def send_item_results(
    batch: list[dict[str, Any]],
    results: list[Any],
    writer_queue: Any,
    worker_id: int,
    device_id: int,
) -> None:
    item_results = []
    for item, result in zip(batch, results):
        analysis_result = extract_analysis_result(result)
        match_score = extract_match_score(result)
        predicted_label = normalize_prediction_label(analysis_result)
        result_dict = result if isinstance(result, dict) else None
        success = result_dict is not None and bool(result_dict.get("success", False))
        # forward bytes for misses AND ambiguous (predicted "ambiguous" — which now also covers
        # "likely_fake" — never equals the true label, so ambiguous-on-fake is included too, and
        # the writer copies it to the miss dir for review even though it counts as correct)
        miss_payload_bytes = None
        if success and predicted_label != item["true_label"]:
            miss_payload_bytes = item["payload_bytes"]

        item_results.append(
            {
                "worker_id": worker_id,
                "device_id": device_id,
                "item": stripped_item(item),
                "result": result,
                "analysis_result": analysis_result,
                "match_score": match_score,
                "predicted_label": predicted_label,
                "miss_payload_bytes": miss_payload_bytes,
            }
        )

    writer_queue.put({"type": "batch_result", "results": item_results})


def process_gpu_batch(
    batch: list[dict[str, Any]],
    inferencer: LocalLivenessBatcher,
    status_queue: Any,
    writer_queue: Any,
    worker_id: int,
    device_id: int,
    fill_elapsed: Optional[float] = None,
) -> int:
    if not batch:
        return 0

    batch_t0 = time.time()
    infer_t0 = time.time()
    try:
        results = inferencer.infer(batch)
    except torch.cuda.OutOfMemoryError as exc:
        infer_elapsed = time.time() - infer_t0
        torch.cuda.empty_cache()
        if len(batch) > 1:
            mid = len(batch) // 2
            left_count = process_gpu_batch(
                batch[:mid],
                inferencer,
                status_queue,
                writer_queue,
                worker_id,
                device_id,
                None,
            )
            right_count = process_gpu_batch(
                batch[mid:],
                inferencer,
                status_queue,
                writer_queue,
                worker_id,
                device_id,
                None,
            )
            return left_count + right_count
        results = [{"success": False, "error": f"CUDA out of memory: {exc}"}]
    except Exception as exc:
        infer_elapsed = time.time() - infer_t0
        if len(batch) > 1:
            mid = len(batch) // 2
            left_count = process_gpu_batch(
                batch[:mid],
                inferencer,
                status_queue,
                writer_queue,
                worker_id,
                device_id,
                None,
            )
            right_count = process_gpu_batch(
                batch[mid:],
                inferencer,
                status_queue,
                writer_queue,
                worker_id,
                device_id,
                None,
            )
            return left_count + right_count
        results = [{"success": False, "error": str(exc)}]
    else:
        infer_elapsed = time.time() - infer_t0

    send_t0 = time.time()
    send_item_results(batch, results, writer_queue, worker_id, device_id)
    result_send_elapsed = time.time() - send_t0
    status_queue.put(
        {
            "type": "batch_done",
            "worker_id": worker_id,
            "device_id": device_id,
            "first_item_id": format_item_id(batch[0]),
            "last_item_id": format_item_id(batch[-1]),
            "batch_size": len(batch),
            "elapsed": time.time() - batch_t0,
            "fill_elapsed": fill_elapsed,
            "infer_elapsed": infer_elapsed,
            "result_send_elapsed": result_send_elapsed,
        }
    )
    return 1


def gpu_worker(
    worker_id: int,
    device_id: int,
    args_dict: dict[str, Any],
    work_queue: Any,
    status_queue: Any,
    writer_queue: Any,
    producers_done_event: Any,
) -> None:
    args = argparse.Namespace(**args_dict)
    batch: list[dict[str, Any]] = []
    total_batches = 0

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
        inferencer = LocalLivenessBatcher(args, device_id, worker_id)
        fill_t0 = time.time()

        while True:
            try:
                queued_work = work_queue.get(timeout=1.0)
            except queue.Empty:
                if producers_done_event.is_set():
                    break
                continue

            if isinstance(queued_work, list):
                batch.extend(queued_work)
            else:
                batch.append(queued_work)

            while len(batch) >= args.batch_size:
                current_batch = batch[: args.batch_size]
                del batch[: args.batch_size]
                fill_elapsed = time.time() - fill_t0
                total_batches += process_gpu_batch(
                    current_batch,
                    inferencer,
                    status_queue,
                    writer_queue,
                    worker_id,
                    device_id,
                    fill_elapsed,
                )
                fill_t0 = time.time()

        if batch:
            fill_elapsed = time.time() - fill_t0
            total_batches += process_gpu_batch(
                batch,
                inferencer,
                status_queue,
                writer_queue,
                worker_id,
                device_id,
                fill_elapsed,
            )

    except Exception as exc:
        status_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "device_id": device_id,
                "error": repr(exc),
            }
        )
    finally:
        try:
            writer_queue.put(
                {
                    "type": "writer_worker_done",
                    "worker_id": worker_id,
                    "device_id": device_id,
                }
            )
        except Exception as exc:
            status_queue.put(
                {
                    "type": "worker_warning",
                    "worker_id": worker_id,
                    "device_id": device_id,
                    "message": f"failed to notify writer: {exc!r}",
                }
            )
        status_queue.put(
            {
                "type": "worker_done",
                "worker_id": worker_id,
                "device_id": device_id,
                "batches": total_batches,
            }
        )


def put_work_item(work_queue: Any, item: Any, stop_event: Any) -> bool:
    while not stop_event.is_set():
        try:
            work_queue.put(item, timeout=1.0)
            return True
        except queue.Full:
            continue
    return False


def safe_queue_size(work_queue: Any) -> Optional[int]:
    try:
        return int(work_queue.qsize())
    except (AttributeError, NotImplementedError, OSError):
        return None


def print_queue_state(
    work_queue: Any,
    queue_slots: int,
    queue_item_capacity: int,
    batch_size: int,
    counts: dict[str, int],
    producer_done: bool,
    worker_done: int,
    worker_count: int,
) -> None:
    approx_size = safe_queue_size(work_queue)
    if approx_size is None:
        batch_text = "unknown"
        item_text = "unknown"
        fullness_text = "unknown"
        state = "unknown"
    else:
        fullness = approx_size / queue_slots if queue_slots > 0 else 0.0
        approx_items = min(approx_size * batch_size, queue_item_capacity)
        batch_text = f"{approx_size}/{queue_slots}"
        item_text = f"~{approx_items}/{queue_item_capacity}"
        fullness_text = f"{fullness * 100.0:.1f}%"
        state = "FULL" if approx_size >= queue_slots else "not_full"

    print(
        f"[QUEUE] fullness={fullness_text} "
        f"processed={counts['total_items']} batches={counts['total_batches']}",
        flush=True,
    )


def producer_loop(
    args: argparse.Namespace,
    producer_id: int,
    media_paths: list[Path],
    input_dir: Path,
    miss_dir: Path,
    work_queue: Any,
    result_queue: Any,
    writer_queue: Any,
    stop_event: Any,
) -> None:
    next_index = 0
    produced_items = 0
    pending_batch: list[dict[str, Any]] = []
    face_filter: Any = None

    def get_face_filter():
        # built once per producer process, only when first needed (insightface on CPU)
        nonlocal face_filter
        if face_filter is None:
            face_filter = FaceQualityFilter(
                args.face_score, args.face_pose, args.face_hmin,
                args.face_hmax, args.face_minpx, args.face_maxside,
            )
        return face_filter

    def enqueue_produced_item(item: dict[str, Any]) -> bool:
        pending_batch.append(item)
        if len(pending_batch) < args.batch_size:
            return True
        batch_to_send = list(pending_batch)
        pending_batch.clear()
        return put_work_item(work_queue, batch_to_send, stop_event)

    def flush_pending_batch() -> bool:
        if not pending_batch:
            return True
        batch_to_send = list(pending_batch)
        pending_batch.clear()
        return put_work_item(work_queue, batch_to_send, stop_event)

    try:
        if not load_image_dependencies():
            raise RuntimeError("failed to load OpenCV/numpy image dependencies")

        for source_path in media_paths:
            if stop_event.is_set():
                break

            true_label = get_true_label_from_path(source_path)
            if true_label is None:
                result_queue.put(
                    {
                        "type": "source_skip",
                        "producer_id": producer_id,
                        "source_path": str(source_path),
                        "reason": "no /real/ or /fake/ in path",
                    }
                )
                continue

            source_suffix = source_path.suffix.lower()
            output_path = get_source_output_path(source_path)
            if args.skip_existing == 1 and has_nonempty_output(output_path):
                result_queue.put(
                    {
                        "type": "source_skip",
                        "producer_id": producer_id,
                        "source_path": str(source_path),
                        "reason": f"{output_path.name} already exists and is non-empty",
                    }
                )
                continue

            if source_suffix in IMAGE_EXTENSIONS:
                next_index += 1
                produced_items += 1
                item_meta = {
                    "index": next_index,
                    "producer_id": producer_id,
                    "kind": "image",
                    "display_path": str(source_path),
                    "source_path": str(source_path),
                    "true_label": true_label,
                    "output_path": str(output_path),
                    "relative_parent": str(relative_parent_for_source(source_path, input_dir)),
                    "subset": f"{true_label}/{subset_for(source_path)}",
                    "miss_filename": (
                        f"{infer_source_name(source_path, input_dir)}_{source_path.stem}.jpg"
                    ),
                }

                try:
                    payload_bytes = source_path.read_bytes()
                except Exception as exc:
                    result_queue.put(
                        {
                            "type": "item_io_fail",
                            "producer_id": producer_id,
                            "item": item_meta,
                            "error": str(exc),
                        }
                    )
                    continue

                # real face-quality filter: skip low-quality REAL images (excluded from accuracy)
                if args.filter_real and true_label == "real":
                    bgr = cv2.imdecode(
                        np.frombuffer(payload_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if bgr is None:
                        result_queue.put(
                            {
                                "type": "item_io_fail",
                                "producer_id": producer_id,
                                "item": item_meta,
                                "error": "failed to decode image for face filter",
                            }
                        )
                        continue
                    keep, reason = get_face_filter().passes(bgr)
                    if not keep:
                        writer_queue.put(
                            {
                                "type": "quality_skip_result",
                                "producer_id": producer_id,
                                "item": item_meta,
                                "reason": reason,
                            }
                        )
                        continue

                item = dict(item_meta)
                item["payload_bytes"] = payload_bytes
                if not enqueue_produced_item(item):
                    break
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                result_queue.put(
                    {
                        "type": "source_fail",
                        "producer_id": producer_id,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "error": "failed to open video",
                    }
                )
                continue

            frame_index = 0
            relative_parent = relative_parent_for_source(source_path, input_dir)
            video_name = source_path.stem

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                next_index += 1
                produced_items += 1
                item_meta = {
                    "index": next_index,
                    "producer_id": producer_id,
                    "kind": "video_frame",
                    "display_path": f"{source_path}#frame={frame_index:06d}",
                    "source_path": str(source_path),
                    "true_label": true_label,
                    "output_path": str(output_path),
                    "frame_index": frame_index,
                    "relative_parent": str(relative_parent),
                    "subset": f"{true_label}/{subset_for(source_path)}",
                    "miss_filename": f"{video_name}_{frame_index:06d}.jpg",
                }

                # real face-quality filter: skip low-quality REAL frames (excluded from accuracy)
                if args.filter_real and true_label == "real":
                    keep, reason = get_face_filter().passes(frame)
                    if not keep:
                        writer_queue.put(
                            {
                                "type": "quality_skip_result",
                                "producer_id": producer_id,
                                "item": item_meta,
                                "reason": reason,
                            }
                        )
                        continue

                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
                )
                if not success:
                    result_queue.put(
                        {
                            "type": "item_io_fail",
                            "producer_id": producer_id,
                            "item": item_meta,
                            "error": "failed to encode frame as JPEG",
                        }
                    )
                    continue

                item = dict(item_meta)
                item["payload_bytes"] = encoded.tobytes()
                if not enqueue_produced_item(item):
                    break

            cap.release()

            if frame_index == 0 and not stop_event.is_set():
                result_queue.put(
                    {
                        "type": "source_fail",
                        "producer_id": producer_id,
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "error": "no frames extracted",
                    }
                )

    except Exception as exc:
        result_queue.put(
            {
                "type": "producer_error",
                "producer_id": producer_id,
                "error": repr(exc),
            }
        )
    finally:
        if not stop_event.is_set():
            flush_pending_batch()
        result_queue.put(
            {
                "type": "producer_done",
                "producer_id": producer_id,
                "produced_items": produced_items,
                "last_index": next_index,
            }
        )


def increment_item_counts(
    item: dict[str, Any],
    label_stats: dict[str, dict[str, int]],
    counts: dict[str, int],
) -> None:
    true_label = item["true_label"]
    if true_label in label_stats:
        label_stats[true_label]["found"] += 1
        label_stats[true_label]["processed"] += 1

    counts["total_items"] += 1
    if item["kind"] == "image":
        counts["still_images_processed"] += 1
    elif item["kind"] == "video_frame":
        counts["video_frames_extracted"] += 1


def handle_item_io_fail(
    message: dict[str, Any],
    label_stats: dict[str, dict[str, int]],
    counts: dict[str, int],
) -> None:
    item = message["item"]
    error_message = message["error"]
    increment_item_counts(item, label_stats, counts)
    counts["failed"] += 1
    label_stats[item["true_label"]]["failed"] += 1
    save_failure_result_for_item(item, error_message)
    print(
        f"[FAIL] item={item['index']} {item['display_path']} error={error_message}",
        file=sys.stderr,
    )


def handle_item_result(
    message: dict[str, Any],
    miss_dir: Path,
    label_stats: dict[str, dict[str, int]],
    counts: dict[str, int],
    print_ok: bool,
    results_fh: Any = None,
    subtally: Optional[dict] = None,
) -> None:
    item = message["item"]
    result = message["result"]
    analysis_result = message["analysis_result"]
    match_score = message["match_score"]
    predicted_label = message["predicted_label"]

    increment_item_counts(item, label_stats, counts)

    if not isinstance(result, dict) or not result.get("success", False):
        save_result_for_item(item, result, analysis_result, predicted_label, match_score, None)
        counts["failed"] += 1
        label_stats[item["true_label"]]["failed"] += 1
        error_message = "N/A"
        if isinstance(result, dict):
            error_message = str(result.get("error", "N/A"))
        shown_result = analysis_result if analysis_result is not None else "N/A"
        shown_match_score = match_score if match_score is not None else "N/A"
        print(
            f"[FAIL] gpu={message['device_id']} item={item['index']} "
            f"{item['display_path']} error={error_message} analysis={shown_result} "
            f"match_score={shown_match_score} predicted={predicted_label} "
            f"true={item['true_label']}",
            file=sys.stderr,
        )
        if results_fh is not None:
            results_fh.write(results_line(
                "ER", item["true_label"], "----", "error", "------", "------", item["display_path"]))
        return

    label_stats[item["true_label"]]["evaluated"] += 1
    miss_output_path = None
    # both "ambiguous" and "likely_fake" normalize to predicted_label "ambiguous", which counts
    # as fake for accuracy (mirrors the ensemble):
    #   * on a FAKE item -> correct, but still copied to the miss dir (flagged "_ambiguous")
    #   * on a REAL item -> miss (copied, flagged "_ambiguous")
    is_ambiguous_like = predicted_label == "ambiguous"
    pred_eval = "fake" if predicted_label in ("fake", "ambiguous") else "real"
    if pred_eval == item["true_label"]:
        counts["correct"] += 1
        label_stats[item["true_label"]]["correct"] += 1
        state = "OK"
    else:
        counts["incorrect"] += 1
        label_stats[item["true_label"]]["miss"] += 1
        state = "MISS"

    # copy to the miss dir for true misses AND for ambiguous/likely_fake (even when correct);
    # ambiguous-like copies get "_ambiguous" inserted into the filename.
    if state == "MISS" or is_ambiguous_like:
        try:
            miss_payload_bytes = message.get("miss_payload_bytes")
            if miss_payload_bytes is None:
                raise RuntimeError("worker did not return missed image bytes")
            miss_item = item
            if is_ambiguous_like:
                mf = item["miss_filename"]
                stem, dot, ext = mf.rpartition(".")
                miss_item = {
                    **item,
                    "miss_filename": f"{stem}_ambiguous.{ext}" if dot else f"{mf}_ambiguous",
                }
            miss_output_path = save_missed_image(miss_item, miss_dir, miss_payload_bytes)
            counts["miss_saved"] += 1
        except Exception as exc:
            print(
                f"[MISS-SAVE-FAIL] item={item['index']} {item['display_path']} error={exc}",
                file=sys.stderr,
            )

    save_result_for_item(
        item,
        result,
        analysis_result,
        predicted_label,
        match_score,
        miss_output_path,
    )

    shown_result = analysis_result if analysis_result is not None else "N/A"
    shown_match_score = match_score if match_score is not None else "N/A"
    extra = f" miss_copy={miss_output_path}" if miss_output_path is not None else ""
    if state != "OK" or print_ok:
        print(
            f"[{state}] gpu={message['device_id']} item={item['index']} "
            f"{item['display_path']} true={item['true_label']} analysis={shown_result} "
            f"match_score={shown_match_score} predicted={predicted_label}{extra}"
        )

    if results_fh is not None:
        tag = "OK" if state == "OK" else "XX"
        ftype = str(analysis_result).strip().lower() if analysis_result is not None else "unknown"
        try:
            match_f = float(match_score) if match_score is not None else None
        except (TypeError, ValueError):
            match_f = None
        if match_f is None:
            fake_disp, match_disp = "------", "------"
        else:
            fake_disp = match_f if ftype == "fake" else (1.0 - match_f)
            match_disp = match_f
        results_fh.write(results_line(
            tag, item["true_label"], predicted_label, ftype, fake_disp, match_disp,
            item["display_path"]))
    if subtally is not None:
        sub = item.get("subset", f"{item['true_label']}/{item['true_label']}")
        st = subtally.setdefault(
            sub, {"group": item["true_label"], "n": 0, "correct": 0, "skipped": 0})
        st["n"] += 1
        if state == "OK":
            st["correct"] += 1


def writer_loop(
    writer_queue: Any,
    status_queue: Any,
    miss_dir_string: str,
    worker_count: int,
    print_ok: bool,
    results_eval_path: str,
) -> None:
    miss_dir = Path(miss_dir_string)
    done_worker_ids: set[int] = set()
    subtally: dict[str, dict] = {}

    os.makedirs(os.path.dirname(os.path.abspath(results_eval_path)) or ".", exist_ok=True)
    results_fh = open(results_eval_path, "w", encoding="utf-8")

    try:
        if not load_image_dependencies():
            raise RuntimeError("failed to load OpenCV/numpy image dependencies")

        while len(done_worker_ids) < worker_count:
            message = writer_queue.get()
            message_type = message.get("type")

            if message_type == "batch_result":
                batch_counts = build_counts()
                batch_label_stats = build_label_stats()
                for item_message in message["results"]:
                    handle_item_result(
                        item_message,
                        miss_dir,
                        batch_label_stats,
                        batch_counts,
                        print_ok,
                        results_fh,
                        subtally,
                    )
                status_queue.put(
                    {
                        "type": "writer_stats_delta",
                        "counts": batch_counts,
                        "label_stats": batch_label_stats,
                    }
                )
            elif message_type == "quality_skip_result":
                skip_item = message["item"]
                skip_reason = message["reason"]
                skip_label = skip_item["true_label"]
                try:
                    save_quality_skip_for_item(skip_item, skip_reason)
                except Exception as exc:
                    print(
                        f"[SK-SAVE-FAIL] {skip_item['display_path']} error={exc}",
                        file=sys.stderr,
                    )
                results_fh.write(results_line(
                    "SK", skip_label, "skip", skip_reason, "------", "------",
                    skip_item["display_path"]))
                sub = skip_item.get("subset", f"{skip_label}/{skip_label}")
                st = subtally.setdefault(
                    sub, {"group": skip_label, "n": 0, "correct": 0, "skipped": 0})
                st["skipped"] += 1
                skip_counts = build_counts()
                skip_counts["real_skipped_quality"] = 1
                skip_label_stats = build_label_stats()
                if skip_label in skip_label_stats:
                    skip_label_stats[skip_label]["skipped_quality"] = 1
                status_queue.put(
                    {
                        "type": "writer_stats_delta",
                        "counts": skip_counts,
                        "label_stats": skip_label_stats,
                    }
                )
                if print_ok:
                    print(f"[SK] {skip_item['display_path']} reason={skip_reason}")
            elif message_type == "writer_worker_done":
                done_worker_ids.add(int(message.get("worker_id", -1)))
            else:
                status_queue.put(
                    {
                        "type": "writer_warning",
                        "message": f"unknown writer message: {message}",
                    }
                )
    except Exception as exc:
        status_queue.put({"type": "writer_error", "error": repr(exc)})
    finally:
        try:
            results_fh.close()
        except Exception:
            pass
        status_queue.put(
            {
                "type": "writer_done",
                "workers_done": len(done_worker_ids),
                "worker_count": worker_count,
                "subtally": subtally,
            }
        )


def handle_source_fail(message: dict[str, Any], counts: dict[str, int]) -> None:
    counts["failed_sources"] += 1
    output_path = Path(message["output_path"])
    append_jsonl(
        output_path,
        {
            "source_video": message["source_path"],
            "error": message["error"],
            "success": False,
        },
    )
    print(
        f"[FAIL-VIDEO] {message['source_path']} error={message['error']}",
        file=sys.stderr,
    )


def build_summary_text(
    args: argparse.Namespace,
    devices: list[int],
    input_dir: Path,
    miss_dir: Path,
    source_image_count: int,
    source_video_count: int,
    counts: dict[str, int],
    label_stats: dict[str, dict[str, int]],
    subtally: dict[str, dict],
) -> str:
    def pct(correct: int, total: int) -> float:
        return (100.0 * correct / total) if total else 0.0

    evaluated = counts["correct"] + counts["incorrect"]
    accuracy = pct(counts["correct"], evaluated)
    mids_path = args.mids_path or os.path.join(args.model_path, "mids.pth")
    filt_desc = "OFF"
    if args.filter_real:
        filt_desc = (f"ON (score>={args.face_score}, pose<={args.face_pose}, "
                     f"hfrac[{args.face_hmin},{args.face_hmax}], minpx={args.face_minpx})")

    lines = [
        "",
        "Summary",
        f"Model path: {args.model_path}",
        f"MIDS checkpoint: {mids_path}",
        f"CUDA devices: {','.join(str(device) for device in devices)}",
        f"Input folder: {input_dir}",
        f"Miss folder: {miss_dir}",
        f"Batch size per GPU: {args.batch_size}",
        f"Real face-quality filter: {filt_desc}",
        f"Source images found: {source_image_count}",
        f"Source videos found: {source_video_count}",
        f"Still images processed: {counts['still_images_processed']}",
        f"Video frames extracted: {counts['video_frames_extracted']}",
        f"Processed items: {counts['total_items']}",
        f"Evaluated: {evaluated}",
        f"Correct: {counts['correct']}",
        f"Incorrect: {counts['incorrect']}",
        f"Miss images saved: {counts['miss_saved']}",
        f"Real skipped (low quality, EXCLUDED from accuracy): {counts['real_skipped_quality']}",
        f"Failed items: {counts['failed']}",
        f"Failed video sources: {counts['failed_sources']}",
        f"Skipped sources: {counts['skipped_sources']}",
        f"Worker errors: {counts['worker_errors']}",
        f"Producer errors: {counts['producer_errors']}",
        f"Writer errors: {counts['writer_errors']}",
        f"Batches processed: {counts['total_batches']}",
        f"Accuracy: {accuracy:.2f}%",
        "",
    ]

    for label in LABELS:
        stats = label_stats[label]
        lines.append(f"{label.capitalize()} count (all found): {stats['found']}")
        lines.append(f"{label.capitalize()} count (processed): {stats['processed']}")
        lines.append(f"{label.capitalize()} count (evaluated): {stats['evaluated']}")
        lines.append(f"{label.capitalize()} accuracy: {pct(stats['correct'], stats['evaluated']):.2f}%")
        lines.append(f"{label.capitalize()} miss: {stats['miss']}")
        lines.append(f"{label.capitalize()} failed: {stats['failed']}")
        lines.append(f"{label.capitalize()} skipped (low quality): {stats['skipped_quality']}")
        lines.append("")

    real_subsets = sorted(s for s in subtally if subtally[s].get("group") == "real")
    fake_subsets = sorted(s for s in subtally if subtally[s].get("group") == "fake")
    if real_subsets:
        lines.append("-- per REAL subset (recall = % correct of kept reals; low-quality excluded) --")
        for s in real_subsets:
            t = subtally[s]
            lines.append(f"   {s:34s} n={t['n']:6d}  recall={pct(t['correct'], t['n']):6.2f}%  "
                         f"skipped={t['skipped']}")
        lines.append("")
    if fake_subsets:
        lines.append("-- per FAKE subset (recall = % flagged fake) --")
        for s in fake_subsets:
            t = subtally[s]
            lines.append(f"   {s:34s} n={t['n']:6d}  recall={pct(t['correct'], t['n']):6.2f}%")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1
    if args.mistral_generations < 1 or args.mistral_generations > 3:
        print("--mistral-generations must be 1, 2, or 3", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required for local checkpoint inference.", file=sys.stderr)
        return 1

    devices = parse_devices(args)
    if not devices:
        print("No CUDA devices selected.", file=sys.stderr)
        return 1
    cuda_count = torch.cuda.device_count()
    invalid_devices = [device for device in devices if device < 0 or device >= cuda_count]
    if invalid_devices:
        print(
            f"Invalid CUDA device ids {invalid_devices}; visible device count is {cuda_count}.",
            file=sys.stderr,
        )
        return 1

    if not load_image_dependencies():
        return 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a folder: {input_dir}", file=sys.stderr)
        return 1

    miss_dir = Path(args.miss_dir).expanduser().resolve()

    media_paths = []
    for path in iter_media_files(input_dir):
        if is_relative_to(path, miss_dir):
            continue
        media_paths.append(path)

    if not media_paths:
        print(f"No image/video files found under {input_dir}")
        return 0

    source_image_count = sum(1 for path in media_paths if path.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"source_image_count:{source_image_count}")
    source_video_count = sum(1 for path in media_paths if path.suffix.lower() in VIDEO_EXTENSIONS)
    print(f"source_video_count:{source_video_count}")

    queue_item_capacity = args.queue_size
    if queue_item_capacity <= 0:
        queue_item_capacity = max(args.batch_size * len(devices) * 16, len(devices) * 16)
    queue_slots = max(1, (queue_item_capacity + args.batch_size - 1) // args.batch_size)

    producer_worker_count = args.producer_workers
    if producer_worker_count <= 0:
        producer_worker_count = min(
            len(devices),
            len(media_paths),
            max(1, os.cpu_count() or 1),
        )
    producer_worker_count = max(1, min(producer_worker_count, len(media_paths)))
    media_path_shards = [
        media_paths[producer_id::producer_worker_count]
        for producer_id in range(producer_worker_count)
    ]

    print(
        f"[INFO] Using {len(devices)} GPU worker(s): {','.join(str(device) for device in devices)}"
    )
    print(f"[INFO] Using {producer_worker_count} producer process(es)")
    print(
        f"[INFO] Work queue capacity: ~{queue_item_capacity} items "
        f"({queue_slots} queued batch slots)"
    )

    ctx = mp.get_context("spawn")
    work_queue = ctx.Queue(maxsize=queue_slots)
    status_queue = ctx.Queue()
    writer_queue = ctx.Queue()
    stop_event = ctx.Event()
    producers_done_event = ctx.Event()

    # consolidated results: the writer owns the per-item results file (OK/XX/SK/ER lines)
    results_eval_path = f"{args.results}.eval"
    os.makedirs(os.path.dirname(os.path.abspath(args.results)) or ".", exist_ok=True)
    subtally: dict[str, dict] = {}

    writer_process = ctx.Process(
        target=writer_loop,
        args=(
            writer_queue,
            status_queue,
            str(miss_dir),
            len(devices),
            args.print_ok,
            results_eval_path,
        ),
    )
    writer_process.start()

    workers = []
    args_dict = vars(args)
    for worker_id, device_id in enumerate(devices):
        process = ctx.Process(
            target=gpu_worker,
            args=(
                worker_id,
                device_id,
                args_dict,
                work_queue,
                status_queue,
                writer_queue,
                producers_done_event,
            ),
        )
        process.start()
        workers.append(process)

    producers = []
    for producer_id, media_path_shard in enumerate(media_path_shards):
        process = ctx.Process(
            target=producer_loop,
            args=(
                args,
                producer_id,
                media_path_shard,
                input_dir,
                miss_dir,
                work_queue,
                status_queue,
                writer_queue,
                stop_event,
            ),
        )
        process.start()
        producers.append(process)

    label_stats = build_label_stats()
    counts = build_counts()

    producer_done_ids: set[int] = set()
    worker_done = 0
    writer_done = False
    exit_code = 0
    last_queue_report_time = time.monotonic()
    producers_joined = False

    while (
        worker_done < len(workers)
        or len(producer_done_ids) < len(producers)
        or not writer_done
    ):
        try:
            message = status_queue.get(timeout=1.0)
        except queue.Empty:
            for producer_id, producer in enumerate(producers):
                if producer_id in producer_done_ids or producer.exitcode is None:
                    continue
                producer_done_ids.add(producer_id)
                if producer.exitcode != 0:
                    counts["producer_errors"] += 1
                    exit_code = 2
                    stop_event.set()
                    print(
                        f"[PRODUCER-ERROR] producer={producer_id} exited with "
                        f"code {producer.exitcode}",
                        file=sys.stderr,
                    )
            if len(producer_done_ids) == len(producers) and not producers_joined:
                for producer in producers:
                    producer.join()
                    if producer.exitcode not in (0, None):
                        exit_code = 2
                producers_done_event.set()
                producers_joined = True
            if not writer_done and writer_process.exitcode is not None:
                writer_done = True
                if writer_process.exitcode != 0:
                    counts["writer_errors"] += 1
                    exit_code = 2
                    print(
                        f"[WRITER-ERROR] writer exited with code {writer_process.exitcode}",
                        file=sys.stderr,
                    )
            if (
                args.queue_report_interval > 0
                and time.monotonic() - last_queue_report_time >= args.queue_report_interval
            ):
                print_queue_state(
                    work_queue,
                    queue_slots,
                    queue_item_capacity,
                    args.batch_size,
                    counts,
                    len(producer_done_ids) == len(producers),
                    worker_done,
                    len(workers),
                )
                last_queue_report_time = time.monotonic()
            continue

        message_type = message.get("type")

        if message_type == "writer_stats_delta":
            merge_counts(counts, message.get("counts", {}))
            merge_label_stats(label_stats, message.get("label_stats", {}))
        elif message_type == "writer_done":
            writer_done = True
            merge_subtally(subtally, message.get("subtally", {}))
            print(
                f"[WRITER-DONE] workers_done={message['workers_done']}/"
                f"{message['worker_count']}"
            )
        elif message_type == "writer_error":
            counts["writer_errors"] += 1
            exit_code = 2
            stop_event.set()
            print(f"[WRITER-ERROR] {message['error']}", file=sys.stderr)
        elif message_type == "writer_warning":
            print(f"[WRITER-WARN] {message['message']}", file=sys.stderr)
        elif message_type == "item_result":
            handle_item_result(message, miss_dir, label_stats, counts, args.print_ok)
        elif message_type == "item_io_fail":
            handle_item_io_fail(message, label_stats, counts)
        elif message_type == "source_skip":
            counts["skipped_sources"] += 1
            print(f"[SKIP] {message['source_path']} ({message['reason']})")
        elif message_type == "source_fail":
            handle_source_fail(message, counts)
        elif message_type == "batch_done":
            counts["total_batches"] += 1
            print(
                f"[BATCH] gpu={message['device_id']} "
                f"count={message['batch_size']} "
                f"first={message['first_item_id']} "
                f"last={message['last_item_id']} "
                f"fill={message.get('fill_elapsed') or 0.0:.2f}s "
                f"infer={message.get('infer_elapsed') or 0.0:.2f}s "
                f"send={message.get('result_send_elapsed') or 0.0:.2f}s "
                f"time={message['elapsed']:.2f}s"
            )
        elif message_type == "worker_error":
            counts["worker_errors"] += 1
            exit_code = 2
            print(
                f"[WORKER-ERROR] gpu={message['device_id']} "
                f"worker={message['worker_id']} error={message['error']}",
                file=sys.stderr,
            )
        elif message_type == "worker_done":
            worker_done += 1
            print(
                f"[WORKER-DONE] gpu={message['device_id']} "
                f"worker={message['worker_id']} batches={message['batches']}"
            )
            if worker_done == len(workers) and len(producer_done_ids) < len(producers):
                stop_event.set()
        elif message_type == "worker_warning":
            print(
                f"[WORKER-WARN] gpu={message['device_id']} "
                f"worker={message['worker_id']} {message['message']}",
                file=sys.stderr,
            )
        elif message_type == "producer_error":
            counts["producer_errors"] += 1
            exit_code = 2
            stop_event.set()
            print(
                f"[PRODUCER-ERROR] producer={message.get('producer_id')} "
                f"{message['error']}",
                file=sys.stderr,
            )
        elif message_type == "producer_done":
            producer_id = int(message.get("producer_id", 0))
            producer_done_ids.add(producer_id)
            print(
                f"[PRODUCER-DONE] producer={producer_id} "
                f"produced_items={message['produced_items']} "
                f"last_index={message['last_index']}"
            )
            if len(producer_done_ids) == len(producers) and not producers_joined:
                for producer in producers:
                    producer.join()
                    if producer.exitcode not in (0, None):
                        exit_code = 2
                producers_done_event.set()
                producers_joined = True
        else:
            print(f"[WARN] Unknown message from worker: {message}", file=sys.stderr)

        if (
            args.queue_report_interval > 0
            and time.monotonic() - last_queue_report_time >= args.queue_report_interval
        ):
            print_queue_state(
                work_queue,
                queue_slots,
                queue_item_capacity,
                args.batch_size,
                counts,
                len(producer_done_ids) == len(producers),
                worker_done,
                len(workers),
            )
            last_queue_report_time = time.monotonic()

    stop_event.set()
    if not producers_joined:
        for producer in producers:
            producer.join()
            if producer.exitcode not in (0, None):
                exit_code = 2
        producers_done_event.set()
        producers_joined = True
    for process in workers:
        process.join()
        if process.exitcode not in (0, None):
            exit_code = 2
    writer_process.join()
    if writer_process.exitcode not in (0, None):
        exit_code = 2

    # drain any late status messages (notably the writer_done that carries the subtally)
    while True:
        try:
            message = status_queue.get_nowait()
        except queue.Empty:
            break
        msg_type = message.get("type")
        if msg_type == "writer_done":
            merge_subtally(subtally, message.get("subtally", {}))
        elif msg_type == "writer_stats_delta":
            merge_counts(counts, message.get("counts", {}))
            merge_label_stats(label_stats, message.get("label_stats", {}))

    summary = build_summary_text(
        args,
        devices,
        input_dir,
        miss_dir,
        source_image_count,
        source_video_count,
        counts,
        label_stats,
        subtally,
    )
    print(summary)

    # consolidated results file: header + per-item lines (OK/XX/SK/ER) + summary
    try:
        with open(args.results, "w", encoding="utf-8") as out:
            out.write(
                f"# FFAA MLLM+MIDS batch test | model={args.model_path} "
                f"| filter_real={bool(args.filter_real)} | input={input_dir} "
                f"| fake_score = match if analysis==fake else 1-match\n"
            )
            out.write(
                "# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image"
                "   (SK = low-quality real, excluded from accuracy)\n"
            )
            if os.path.exists(results_eval_path):
                with open(results_eval_path, encoding="utf-8") as fh:
                    out.write(fh.read())
            out.write(summary + "\n")
        try:
            os.remove(results_eval_path)
        except OSError:
            pass
        print(f"\n[wrote consolidated per-item results + summary -> {args.results}]")
    except Exception as exc:
        print(f"[RESULTS-WRITE-FAIL] {args.results} error={exc}", file=sys.stderr)

    if (
        counts["failed"]
        or counts["failed_sources"]
        or counts["worker_errors"]
        or counts["producer_errors"]
        or counts["writer_errors"]
    ):
        return 2 if exit_code == 0 else exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
