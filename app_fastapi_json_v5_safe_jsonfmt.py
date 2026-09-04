"""PAAS Face Personhood API (FastAPI) -- backed by the PAAS fused ensemble.

This server now scores every request with the unified ``PaasPipeline`` (FFAA MLLM+MIDS fused with the
MLLM-free 9-class ensemble), instead of the standalone FFAA model it used previously. A single
pipeline is loaded at startup and shared by all endpoints.

Endpoints:
  * POST /face_liveness                 -- one uploaded image           -> fused face_liveness JSON
  * POST /face_liveness_base64          -- one base64 image             -> fused face_liveness JSON
  * POST /face_liveness_base64_batch    -- N base64 images              -> per-image fused results
  * POST /face_liveness_new             -- exactly 5 uploaded images     -> ONE final verdict (NEW)
  * POST /face_liveness_base64_batch_new-- exactly 5 base64 images       -> ONE final verdict (NEW)

The two *_new endpoints take 5 frames of the same subject, score each with the fused model, drop the
single highest and single lowest fake-score, average the remaining three (trimmed mean), and decide
the final judgement from that aggregate -- robust to one fluky frame in either direction.
"""
import base64
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from io import BytesIO
from collections import Counter
from concurrent.futures import Future, TimeoutError
from queue import Queue, Empty
from threading import Thread
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Pin the GPU before any CUDA/transformers import, then make the vendored ffaa/ + ensemble9/ trees
# importable (env.setup also sets HF offline + the device). This must precede the ffaa imports below.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paas import env
env.setup("cuda:0")

import torch  # noqa: E402  (after env.setup so CUDA_VISIBLE_DEVICES is pinned)
from paas.config import PaasConfig  # noqa: E402
from paas.pipeline import PaasPipeline  # noqa: E402
from paas.decision import decide  # noqa: E402
from utils.file_utils import get_jsonfmt  # noqa: E402  (vendored ffaa util, importable post env.setup)
from yolo11_cls_onnx import Yolo11ClsONNX  # noqa: E402

from fastapi import Body, FastAPI, File, UploadFile, Request, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, HTMLResponse  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

# ----------------------------------------------------------------------------- config / constants
APP_ROOT = _HERE
PAAS_CONFIG = os.environ.get("PAAS_CONFIG", os.path.join("config", "experiments", "paas5_mean.json"))
# resolve a relative config against the app dir so the server runs from ANY working directory
if not os.path.isabs(PAAS_CONFIG):
    PAAS_CONFIG = os.path.join(APP_ROOT, PAAS_CONFIG)
ENS_BATCH = int(os.environ.get("ENS_BATCH", "32"))
FFAA_BATCH = int(os.environ.get("FFAA_BATCH", "8"))
# number of frames the *_new endpoints expect (trimmed mean drops 1 highest + 1 lowest of these)
NEW_BATCH_SIZE = int(os.environ.get("NEW_BATCH_SIZE", "5"))

IMAGE_FORMAT_TO_EXT = {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
FORGERY_LABEL = {"real": "None", "pad": "PAD (presentation attack / spoof)",
                 "deepfake": "Deepfake", "fake": "Fake"}

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
DYNAMIC_BATCHING = os.environ.get("DYNAMIC_BATCHING", "1") == "1"
DYNAMIC_BATCH_MAX_WAIT_MS = int(os.environ.get("DYNAMIC_BATCH_MAX_WAIT_MS", "120"))
INFERENCE_TIMEOUT_SEC = int(os.environ.get("INFERENCE_TIMEOUT_SEC", "180"))
IGNORE_ROTATION_CHECK = os.environ.get("IGNORE_ROTATION_CHECK", "1") == "1"
SAVE_INPUT_IMAGES = os.environ.get("SAVE_INPUT_IMAGES", "1") == "1"
TEMP_IMAGE_DIR = os.environ.get("TEMP_IMAGE_DIR", None)

# ----------------------------------------------------------------------------- model + rotation det
det_model_path = os.environ.get(
    "ROT_DET_MODEL", os.path.join(env.FFAA_DIR, "rot_det_model", "yolo11n-rotation2", "weights", "best.onnx"))
det_model = Yolo11ClsONNX(onnx_path=det_model_path, imgsz=224, class_names=["0", "180", "270", "90"])

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("ALLOW_TF32", "1") == "1"
    torch.backends.cudnn.allow_tf32 = os.environ.get("ALLOW_TF32", "1") == "1"

GSD_BATCH = int(os.environ.get("GSD_BATCH", "32"))
SELOP_BATCH = int(os.environ.get("SELOP_BATCH", "32"))

print(f"[app] loading PAAS pipeline from {PAAS_CONFIG} ...", flush=True)
_cfg = PaasConfig.from_file(PAAS_CONFIG)
_cfg.device = "cuda:0"
# PAAS_COMPONENTS lets a deployment pick the combination without editing the config, e.g.
#   PAAS_COMPONENTS="A1_9c,A2_9c,gsd,selop"  -> fast, no 7B MLLM (see config/experiments/paas4_fast.json)
if os.environ.get("PAAS_COMPONENTS"):
    _cfg.fusion.components = [c.strip() for c in os.environ["PAAS_COMPONENTS"].split(",") if c.strip()]
if os.environ.get("PAAS_FUSION"):
    _cfg.fusion.method = os.environ["PAAS_FUSION"]
if os.environ.get("PAAS_THRESHOLD"):
    _cfg.decision.threshold = float(os.environ["PAAS_THRESHOLD"])
# load only the detectors the requested components need
_need = _cfg.needs()
_cfg.ffaa.enabled = _need["ffaa"]; _cfg.ensemble9.enabled = _need["ens"]
_cfg.gsd.enabled = _need["gsd"]; _cfg.selop.enabled = _need["selop"]
_cfg.validate()
pipe = PaasPipeline(_cfg)
TAU = float(_cfg.decision.threshold)
REAL_AMB_MIN = float(_cfg.decision.real_ambiguous_match_min)
print(f"[app] pipeline ready: fusion={_cfg.fusion.method} components={_cfg.fusion.components} "
      f"threshold={TAU} | ffaa={pipe.ffaa is not None} ens={pipe.ens is not None} "
      f"gsd={pipe.gsd is not None} selop={pipe.selop is not None}", flush=True)


# ----------------------------------------------------------------------------- FastAPI app
tags_metadata = [
    {"name": "status", "description": "Health, runtime configuration, and batching diagnostics."},
    {"name": "liveness", "description": "Face liveness/personhood analysis endpoints (PAAS fused model)."},
]

app = FastAPI(
    title="PAAS Face Personhood API",
    version=os.environ.get("API_VERSION", "4.0-paas-fused"),
    description=(
        "PAAS face Personhood API backed by the fused FFAA+9-class ensemble, with dynamic GPU request "
        "batching and JSON-object face_liveness responses. Swagger UI at `/docs`, ReDoc at `/redoc`."
    ),
    docs_url=os.environ.get("SWAGGER_DOCS_URL", "/docs"),
    redoc_url=os.environ.get("REDOC_URL", "/redoc"),
    openapi_url=os.environ.get("OPENAPI_URL", "/openapi.json"),
    openapi_tags=tags_metadata,
    swagger_ui_parameters={"displayRequestDuration": True, "filter": True, "tryItOutEnabled": True,
                           "defaultModelsExpandDepth": 1, "defaultModelExpandDepth": 2},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------- request/response models
class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image (data: URL prefix accepted).",
                              examples=["/9j/4AAQSkZJRgABAQAAAQABAAD..."])


class Base64BatchRequest(BaseModel):
    images_base64: Optional[List[str]] = Field(
        default=None, description="List of base64-encoded images. Preferred field name.",
        examples=[["/9j/4AAQSkZJRgABAQAAAQABAAD...", "iVBORw0KGgoAAAANSUhEUg..."]])
    image_base64_list: Optional[List[str]] = Field(
        default=None, description="Backward-compatible alias for images_base64.")


class HealthResponse(BaseModel):
    success: bool
    status: str


class FinalJudgementResponse(BaseModel):
    success: bool = True
    final: Dict[str, Any] = Field(..., description="Aggregated verdict over the 5 frames.")
    aggregate: Dict[str, Any] = Field(..., description="Trimmed-mean details (scores used / excluded).")
    per_image: List[Dict[str, Any]] = Field(..., description="Per-frame fused score + decision.")
    processing_time_ms: float = Field(..., description="Inference processing time for the request, in milliseconds.")


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
        "<html><body><h3>PAAS Face Liveness API (fused)</h3>"
        "<p>FastAPI server is running.</p><ul>"
        "<li><a href='/docs'>Swagger UI</a></li><li><a href='/redoc'>ReDoc</a></li>"
        "<li><a href='/batcher_status'>Batcher status</a></li></ul></body></html>"
    )


@app.get("/health", response_model=HealthResponse, tags=["status"], summary="Health check")
async def health():
    return {"success": True, "status": "ok"}


@app.get("/batcher_status", tags=["status"], summary="Show batching/runtime settings")
async def batcher_status():
    return {
        "dynamic_batching": inference_batcher is not None,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_wait_ms": DYNAMIC_BATCH_MAX_WAIT_MS,
        "queue_size": inference_batcher.queue.qsize() if inference_batcher is not None else 0,
        "fusion": _cfg.fusion.method,
        "components": _cfg.fusion.components,
        "threshold": TAU,
        "ffaa_enabled": pipe.ffaa is not None,
        "ensemble_enabled": pipe.ens is not None,
        "gsd_enabled": pipe.gsd is not None,
        "selop_enabled": pipe.selop is not None,
        "new_batch_size": NEW_BATCH_SIZE,
        "inference_timeout_sec": INFERENCE_TIMEOUT_SEC,
        "ignore_rotation_check": IGNORE_ROTATION_CHECK,
        "save_input_images": SAVE_INPUT_IMAGES,
        "temp_image_dir": TEMP_IMAGE_DIR,
    }


# ----------------------------------------------------------------------------- image I/O helpers
def getRotatedAngle(img):
    if img is None:
        raise ValueError("Failed to read image")
    rot_angle, conf = det_model.predict(img)   # angle in {0,90,180,270}
    return rot_angle


def get_request_image_subdir(request_obj: Request | None = None):
    if request_obj is not None:
        ipaddr = request_obj.headers.get("X-Forwarded-For") or (
            request_obj.client.host if request_obj.client else None)
    else:
        ipaddr = "unknown"
    if ipaddr:
        ipaddr = ipaddr.split(",")[0].strip()
    if not ipaddr:
        ipaddr = "unknown"
    subdir = os.path.join(APP_ROOT, "images", ipaddr)
    try:
        os.makedirs(subdir, exist_ok=True)
        return subdir
    except OSError as exc:
        print(f"warning: failed to create request image dir {subdir}: {exc}")
        return None


def get_image_suffix_from_bytes(image_bytes):
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Failed to decode image") from exc
    return IMAGE_FORMAT_TO_EXT.get(image_format, ".png")


def save_request_image_bytes(image_bytes, suffix, request_obj: Request | None = None):
    """Persist an image for inference. Returns (file_path, should_delete_after_inference)."""
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix
    if SAVE_INPUT_IMAGES:
        subdir = get_request_image_subdir(request_obj)
        if subdir:
            file_path = os.path.join(subdir, fname)
            try:
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                return file_path, False
            except OSError as exc:
                print(f"warning: failed to archive request image to {file_path}: {exc}")
    try:
        if TEMP_IMAGE_DIR:
            os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix="paas_frame_", suffix=suffix, dir=TEMP_IMAGE_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return tmp_path, True
    except OSError as exc:
        print(f"error: failed to save request image to temp dir "
              f"{TEMP_IMAGE_DIR or tempfile.gettempdir()}: {exc}")
        return None, False


def cleanup_temp_paths(paths: List[str]):
    for path in paths:
        if not path:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"warning: failed to delete temp image {path}: {exc}")


def safe_write_text(path: str, text: str) -> bool:
    """Best-effort result save. Never let optional logging crash inference."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except OSError as exc:
        print(f"warning: failed to save result text to {path}: {exc}")
        return False


def save_result_text(image_path: str, result: Any, infix: str = "") -> None:
    """When SAVE_INPUT_IMAGES=1, write the inference result as JSON next to the archived image
    (``<image-stem><infix>.txt``). No-op otherwise (temp images are deleted after inference)."""
    if not SAVE_INPUT_IMAGES or not image_path:
        return
    stem, _ext = os.path.splitext(image_path)
    try:
        safe_write_text(stem + infix + ".txt", json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"warning: failed to serialize result for {image_path}: {exc}")


def decode_base64_image(image_base64: str) -> bytes:
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 must be a non-empty string")
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc


# ----------------------------------------------------------------------------- response formatting
def _normalize_face_liveness_output(output):
    """Make a face_liveness dict safe for JSON clients (fill Nones / missing fields)."""
    if not isinstance(output, dict):
        output = {}
    image_desc = output.get("Image description")
    if not isinstance(image_desc, dict):
        image_desc = {}
    image_desc["focus"] = image_desc.get("focus") or "unknown"
    image_desc["resolution"] = image_desc.get("resolution") or "unknown"
    image_desc["quality"] = image_desc.get("quality") or "unknown"
    output["Image description"] = image_desc
    output["Forgery reasoning"] = output.get("Forgery reasoning") or (
        "Insufficient structured model output; treated conservatively.")
    output["Analysis result"] = output.get("Analysis result") or "ambiguous"
    output["Forgery type"] = output.get("Forgery type") or "ambiguous"
    if output.get("Match score") is None:
        output["Match score"] = ""
    if output.get("Difficulty") is None:
        output["Difficulty"] = "unknown"
    return output


def build_face_liveness_response(face_liveness):
    """Wrap a face_liveness payload as {success, face_liveness}. Accepts either an already-built
    dict (from the pipeline) or a raw FFAA answer string (parsed via get_jsonfmt). Never raises."""
    try:
        output = face_liveness if isinstance(face_liveness, dict) else get_jsonfmt(face_liveness)
        output = _normalize_face_liveness_output(output)
    except Exception as exc:
        print(f"warning: face_liveness formatting failed; conservative ambiguous result: {exc}")
        output = _normalize_face_liveness_output({"Formatter error": str(exc)})
    return {"success": True, "face_liveness": output}


def face_liveness_from_result(r: dict) -> dict:
    """Build the rich face_liveness JSON from a fused pipeline result. The verdict fields come from
    the FUSED decision; Image description / Forgery reasoning are lifted from the FFAA answer text
    when available (else synthesized)."""
    decision = r.get("decision", "ambiguous")
    ftype = r.get("forgery_type", "fake")
    fl: Dict[str, Any] = {}
    answer = r.get("ffaa_answer")
    if answer:
        try:
            j = get_jsonfmt(answer)
            if isinstance(j, dict):
                fl["Image description"] = j.get("Image description")
                fl["Forgery reasoning"] = j.get("Forgery reasoning")
        except Exception:
            pass
    fl["Analysis result"] = decision
    fl["Forgery type"] = FORGERY_LABEL.get(ftype, ftype) if decision != "real" else "None"
    fl["Forgery score"] = f"{r['forgery_score']:.4f}" if r.get("forgery_score") is not None else ""
    fl["Match score"] = f"{r['match_score']:.4f}" if r.get("match_score") is not None else ""
    fl["Model"] = f"PAAS_v3 fused [{_cfg.fusion.method}] over {_cfg.fusion.components}"
    fl["Details"] = {"components": r.get("components"),
                     "ensemble_fake": r.get("ensemble_fake"), "ffaa_fake": r.get("ffaa_fake"),
                     "gsd_fake": r.get("gsd_fake"), "selop_fake": r.get("selop_fake"),
                     "per_model_fake": r.get("ensemble_per_model"), "threshold": TAU}
    return _normalize_face_liveness_output(fl)


# ----------------------------------------------------------------------------- core scoring
def score_paths_batch(image_paths: List[str]) -> List[dict]:
    """Rotation-check + decode each path, score the valid frames with the fused PaasPipeline, and
    return one item per input path (input order). Each item is either an error dict or:
        {success, face_liveness, fused_fake, decision, forgery_type, match_score,
         ensemble_fake, ffaa_fake}
    """
    results: List[Optional[dict]] = [None] * len(image_paths)
    valid_idx, valid_rgb, valid_keys = [], [], []
    for idx, path in enumerate(image_paths):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            results[idx] = {"success": False, "error": "Failed to read image"}
            continue
        try:
            angle = getRotatedAngle(bgr)
        except Exception as exc:
            angle = 0
            print(f"warning: rotation detector failed for {path}: {exc}")
        if angle and angle > 0 and not IGNORE_ROTATION_CHECK:
            results[idx] = {"success": False,
                            "error": "The image appears to be rotated. Please try again with a straightened image."}
            continue
        valid_idx.append(idx)
        valid_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        valid_keys.append(path)

    if valid_rgb:
        scored = pipe.predict_frames(valid_rgb, keys=valid_keys,
                                     ens_batch_size=ENS_BATCH, ffaa_batch_size=FFAA_BATCH,
                                     gsd_batch_size=GSD_BATCH, selop_batch_size=SELOP_BATCH)
        for j, idx in enumerate(valid_idx):
            r = scored[j]
            if r.get("decision") == "error":
                results[idx] = {"success": False, "error": r.get("error", "inference failed")}
                continue
            results[idx] = {
                "success": True,
                "face_liveness": face_liveness_from_result(r),
                "fused_fake": float(r["forgery_score"]),
                "decision": r["decision"],
                "forgery_type": r.get("forgery_type"),
                "match_score": float(r["match_score"]),
                "ensemble_fake": r.get("ensemble_fake"),
                "ffaa_fake": r.get("ffaa_fake"),
                "gsd_fake": r.get("gsd_fake"),
                "selop_fake": r.get("selop_fake"),
                "components": r.get("components"),
            }
    for idx, res in enumerate(results):
        if res is None:
            results[idx] = {"success": False, "error": "Unknown input error"}
    # persist the per-image result next to the archived image (only when SAVE_INPUT_IMAGES=1)
    if SAVE_INPUT_IMAGES:
        for path, res in zip(image_paths, results):
            save_result_text(path, res)
    return results


def aggregate_trimmed(items: List[dict]) -> dict:
    """Trimmed-mean aggregation over per-frame fused fake-scores: drop ONE highest + ONE lowest,
    average the rest, then apply the PAAS decision rule. Returns the final verdict + details."""
    scored = [it for it in items if it.get("success") and it.get("fused_fake") is not None]
    if not scored:
        return {"ok": False, "error": "no valid frames to aggregate"}
    scored_sorted = sorted(scored, key=lambda it: it["fused_fake"])
    if len(scored_sorted) >= 3:
        kept = scored_sorted[1:-1]                      # exclude single lowest + single highest
        excluded = {"lowest": scored_sorted[0]["fused_fake"], "highest": scored_sorted[-1]["fused_fake"]}
    else:
        kept = scored_sorted                            # too few frames to trim -> plain mean
        excluded = {"lowest": None, "highest": None}
    used = [it["fused_fake"] for it in kept]
    agg = float(sum(used) / len(used))

    # forgery type for the aggregate: majority of kept frames' fused types (pad/deepfake)
    types = [it.get("forgery_type") for it in kept if it.get("forgery_type") in ("pad", "deepfake")]
    mode_type = Counter(types).most_common(1)[0][0] if types else None
    d = decide(agg, _cfg.decision, type_probs=None, ffaa_forgery_type=mode_type)

    return {
        "ok": True,
        "decision": d["decision"],
        "forgery_type": d["forgery_type"],
        "forgery_score": d["forgery_score"],
        "match_score": d["match_score"],
        "threshold": TAU,
        "n_total": len(items),
        "n_used": len(used),
        "used_scores": [round(x, 4) for x in used],
        "excluded": {k: (round(v, 4) if v is not None else None) for k, v in excluded.items()},
    }


# ----------------------------------------------------------------------------- dynamic batcher
class DynamicInferenceBatcher:
    """Coalesce concurrent HTTP requests into one GPU batch for higher throughput under load."""

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
            while total < self.max_batch_size:
                try:
                    paths, future = self.queue.get_nowait()
                except Empty:
                    break
                batch_items.append((paths, future))
                total += len(paths)

            flat_paths, slices, cursor = [], [], 0
            for paths, _f in batch_items:
                flat_paths.extend(paths)
                slices.append((cursor, cursor + len(paths)))
                cursor += len(paths)
            print(f"dynamic batcher dispatching: grouped_requests={len(batch_items)}, "
                  f"images={len(flat_paths)}", flush=True)
            try:
                batch_results = score_paths_batch(flat_paths)
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


def score_paths_dynamic(image_paths: List[str]) -> List[dict]:
    if inference_batcher is None:
        return score_paths_batch(image_paths)
    return inference_batcher.submit_many(image_paths)


# ----------------------------------------------------------------------------- shared request plumbing
def _save_uploads(file_bytes_list, suffixes, request) -> tuple:
    """Save a list of image byte-blobs to temp files. Returns (paths, temp_paths_to_delete)."""
    paths, to_delete = [], []
    for b, suffix in zip(file_bytes_list, suffixes):
        p, should_delete = save_request_image_bytes(b, suffix, request)
        if not p:
            raise HTTPException(status_code=500, detail="Failed to save image for inference")
        paths.append(p)
        if should_delete:
            to_delete.append(p)
    return paths, to_delete


async def _run_new_endpoint(image_bytes_list, suffixes, request) -> dict:
    """Shared body for the two *_new endpoints: save -> score -> trimmed-mean verdict."""
    if len(image_bytes_list) != NEW_BATCH_SIZE:
        raise HTTPException(status_code=400,
                            detail=f"exactly {NEW_BATCH_SIZE} images are required (got {len(image_bytes_list)})")
    paths, to_delete = _save_uploads(image_bytes_list, suffixes, request)
    t0 = time.perf_counter()
    try:
        items = await run_in_threadpool(score_paths_dynamic, paths)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")
    finally:
        cleanup_temp_paths(to_delete)
    proc_ms = round((time.perf_counter() - t0) * 1000, 2)

    agg = aggregate_trimmed(items)
    per_image = []
    for i, it in enumerate(items):
        if it.get("success"):
            per_image.append({"index": i, "fused_fake": round(it["fused_fake"], 4),
                              "decision": it["decision"], "forgery_type": it.get("forgery_type"),
                              "match_score": round(it["match_score"], 4),
                              "ensemble_fake": it.get("ensemble_fake"), "ffaa_fake": it.get("ffaa_fake"),
                              "face_liveness": it.get("face_liveness")})
        else:
            per_image.append({"index": i, "success": False, "error": it.get("error")})

    if not agg["ok"]:
        resp = {"success": False, "error": agg["error"], "per_image": per_image,
                "processing_time_ms": proc_ms}
        if paths:
            save_result_text(paths[0], resp, infix=".final")   # group verdict next to 1st frame
        return resp
    final = {
        "Analysis result": agg["decision"],
        "Forgery type": FORGERY_LABEL.get(agg["forgery_type"], agg["forgery_type"])
        if agg["decision"] != "real" else "None",
        "Forgery score": f"{agg['forgery_score']:.4f}",
        "Match score": f"{agg['match_score']:.4f}",
        "Threshold": round(TAU, 4),
        "Frames used": agg["n_used"], "Frames total": agg["n_total"],
        "Model": f"PAAS fused [{_cfg.fusion.method}] -- trimmed-mean over {NEW_BATCH_SIZE} frames",
    }
    aggregate = {"method": "trimmed_mean(drop 1 highest + 1 lowest)",
                 "score": round(agg["forgery_score"], 4),
                 "used_scores": agg["used_scores"], "excluded": agg["excluded"]}
    resp = {"success": True, "final": final, "aggregate": aggregate, "per_image": per_image,
            "processing_time_ms": proc_ms}
    if paths:
        save_result_text(paths[0], resp, infix=".final")       # group verdict next to 1st frame
    return resp


# ----------------------------------------------------------------------------- existing endpoints (now fused)
@app.post("/face_liveness", tags=["liveness"], summary="Analyze one uploaded face image (fused)")
async def receive_face(request: Request, face: UploadFile = File(..., description="Face image file")):
    if not face.filename:
        raise HTTPException(status_code=400, detail="no face image file.")
    image_bytes = await face.read()
    try:
        suffix = get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    file_path, should_delete = save_request_image_bytes(image_bytes, suffix, request)
    if not file_path:
        return JSONResponse({"success": False, "error": "Failed to save image for inference"})
    t0 = time.perf_counter()
    try:
        res = (await run_in_threadpool(score_paths_dynamic, [file_path]))[0]
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")
    finally:
        if should_delete:
            cleanup_temp_paths([file_path])
    proc_ms = round((time.perf_counter() - t0) * 1000, 2)
    if not res.get("success"):
        return JSONResponse({"success": False, "error": res.get("error", "inference failed"),
                             "processing_time_ms": proc_ms})
    out = build_face_liveness_response(res["face_liveness"])
    out["processing_time_ms"] = proc_ms
    return JSONResponse(out)


@app.post("/face_liveness_base64", tags=["liveness"], summary="Analyze one base64 face image (fused)")
async def receive_face_base64(request: Request, payload: Base64ImageRequest = Body(...)):
    try:
        image_bytes = decode_base64_image(payload.image_base64)
        suffix = get_image_suffix_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    img_path, should_delete = save_request_image_bytes(image_bytes, suffix, request)
    if not img_path:
        return JSONResponse({"success": False, "error": "Failed to save image for inference"})
    t0 = time.perf_counter()
    try:
        res = (await run_in_threadpool(score_paths_dynamic, [img_path]))[0]
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")
    finally:
        if should_delete:
            cleanup_temp_paths([img_path])
    proc_ms = round((time.perf_counter() - t0) * 1000, 2)
    if not res.get("success"):
        return JSONResponse({"success": False, "error": res.get("error", "inference failed"),
                             "processing_time_ms": proc_ms})
    out = build_face_liveness_response(res["face_liveness"])
    out["processing_time_ms"] = proc_ms
    return JSONResponse(out)


@app.post("/face_liveness_base64_batch", tags=["liveness"], summary="Analyze a batch of base64 images (fused)")
async def receive_face_base64_batch(request: Request, payload: Base64BatchRequest = Body(...)):
    images_base64 = payload.images_base64 or payload.image_base64_list
    if not isinstance(images_base64, list) or len(images_base64) == 0:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list")
    if len(images_base64) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}")

    results = [None] * len(images_base64)
    valid_idx, valid_paths, to_delete = [], [], []
    for idx, b64 in enumerate(images_base64):
        try:
            image_bytes = decode_base64_image(b64)
            suffix = get_image_suffix_from_bytes(image_bytes)
        except ValueError as exc:
            results[idx] = {"success": False, "error": str(exc)}
            continue
        img_path, should_delete = save_request_image_bytes(image_bytes, suffix, request)
        if not img_path:
            results[idx] = {"success": False, "error": "Failed to save image for inference"}
            continue
        valid_idx.append(idx)
        valid_paths.append(img_path)
        if should_delete:
            to_delete.append(img_path)

    proc_ms = 0.0
    if valid_paths:
        t0 = time.perf_counter()
        try:
            batch = await run_in_threadpool(score_paths_dynamic, valid_paths)
        except Exception as exc:
            batch = [{"success": False, "error": str(exc)} for _ in valid_paths]
        finally:
            cleanup_temp_paths(to_delete)
        proc_ms = round((time.perf_counter() - t0) * 1000, 2)
        for local, original in enumerate(valid_idx):
            it = batch[local]
            results[original] = (build_face_liveness_response(it["face_liveness"])
                                 if it.get("success") else {"success": False, "error": it.get("error")})
    for idx, r in enumerate(results):
        if r is None:
            results[idx] = {"success": False, "error": "Unknown input error"}
    return JSONResponse({"success": True, "results": results, "processing_time_ms": proc_ms})


# ----------------------------------------------------------------------------- NEW: 5-frame endpoints
@app.post("/face_liveness_new", response_model=FinalJudgementResponse, tags=["liveness"],
          summary=f"Analyze {NEW_BATCH_SIZE} uploaded face images -> ONE trimmed-mean verdict")
async def receive_face_new(request: Request,
                           faces: List[UploadFile] = File(..., description=f"{NEW_BATCH_SIZE} face image files")):
    if not faces or any(not f.filename for f in faces):
        raise HTTPException(status_code=400, detail="image files are required")
    image_bytes_list, suffixes = [], []
    for f in faces:
        b = await f.read()
        try:
            suffixes.append(get_image_suffix_from_bytes(b))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        image_bytes_list.append(b)
    return JSONResponse(await _run_new_endpoint(image_bytes_list, suffixes, request))


@app.post("/face_liveness_base64_batch_new", response_model=FinalJudgementResponse, tags=["liveness"],
          summary=f"Analyze {NEW_BATCH_SIZE} base64 face images -> ONE trimmed-mean verdict")
async def receive_face_base64_batch_new(request: Request, payload: Base64BatchRequest = Body(...)):
    images_base64 = payload.images_base64 or payload.image_base64_list
    if not isinstance(images_base64, list) or len(images_base64) == 0:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list")
    image_bytes_list, suffixes = [], []
    for b64 in images_base64:
        try:
            b = decode_base64_image(b64)
            suffixes.append(get_image_suffix_from_bytes(b))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        image_bytes_list.append(b)
    return JSONResponse(await _run_new_endpoint(image_bytes_list, suffixes, request))


# ----------------------------------------------------------------------------- entrypoint
if __name__ == "__main__":
    import uvicorn

    cert_path = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "certs/key.pem")
    ssl_kwargs = {}
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        ssl_kwargs = {"ssl_certfile": cert_path, "ssl_keyfile": key_path}

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), workers=1, **ssl_kwargs)
