#!/usr/bin/env python3.12
"""FastAPI server for PAAS_ensemble_v2 (combined FFAA MLLM+MIDS + 9-class MLLM-free ensemble).

Mirrors the FFAA / MIDS++ `app_fastapi_json.py` request/response conventions (same endpoint names,
status routes, CORS + no-cache middleware, JSON-object responses) but the model is the combined
`PaasPipeline`: it runs whichever models the experiment config selects and fuses their per-frame
fake-scores. The fused `forgery_score` is threshold-independent, so the decision threshold can be
overridden per request without reloading.

Run (global python3.12 / transformers==4.37.2):
    python3.12 app_fastapi_json.py                         # serves on 0.0.0.0:3000
    PAAS_CONFIG=config/experiments/weighted_ffaa.json PORT=3000 DEVICE=cuda:0 python3.12 app_fastapi_json.py
Docs:  http://<host>:3000/docs
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
from datetime import datetime
from io import BytesIO
from typing import List, Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from PIL import Image, UnidentifiedImageError
from fastapi import Body, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
from paas.config import PaasConfig
from paas.pipeline import PaasPipeline
from paas.decision import decide

# --------------------------------------------------------------------------------------
# configuration (overridable via environment)
# --------------------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("PAAS_CONFIG", os.path.join(_HERE, "config", "experiments", "weighted_ffaa.json"))
DEVICE = os.environ.get("DEVICE")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "16"))
FFAA_BATCH = int(os.environ.get("FFAA_BATCH", "8"))
ENS_BATCH = int(os.environ.get("ENS_BATCH", "32"))
SAVE_REQUESTS = os.environ.get("SAVE_REQUESTS", "0") == "1"
REQUEST_DIR = os.environ.get("REQUEST_DIR", os.path.join(_HERE, "request_images"))

IMAGE_FORMAT_TO_EXT = {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
FORGERY_LABEL = {"real": "None", "pad": "PAD (presentation attack / spoof)", "deepfake": "Deepfake"}

# --------------------------------------------------------------------------------------
# load the combined pipeline once at startup
# --------------------------------------------------------------------------------------
cfg = PaasConfig.from_file(CONFIG_PATH)
if DEVICE:
    cfg.device = DEVICE
print(f"[startup] loading PaasPipeline config='{cfg.name}' fusion={cfg.fusion.method} "
      f"(ffaa={cfg.ffaa.enabled}, ensemble9={cfg.ensemble9.enabled}) on {cfg.device} ...", flush=True)
_t = time.time()
engine = PaasPipeline(cfg)
print(f"[startup] loaded in {time.time()-_t:.1f}s "
      f"(ffaa={engine.ffaa is not None}, ensemble9={engine.ens is not None})", flush=True)
os.makedirs(REQUEST_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------
# scoring + response
# --------------------------------------------------------------------------------------
def score_paths(paths: List[str]) -> List[dict]:
    return engine.predict_images(paths, ens_batch_size=ENS_BATCH, ffaa_batch_size=FFAA_BATCH)


def build_response(r: dict, threshold: Optional[float]) -> dict:
    """Re-decide from the (threshold-independent) fused forgery_score so threshold can be per-request."""
    ff = float(r["forgery_score"])
    tau = cfg.decision.threshold if threshold is None else float(threshold)
    # reuse the project's decision rule; carry forgery_type from the original (pad vs deepfake).
    dcfg = type(cfg.decision)(threshold=tau,
                              real_ambiguous_match_min=cfg.decision.real_ambiguous_match_min,
                              treat_likely_fake_as_ambiguous=cfg.decision.treat_likely_fake_as_ambiguous)
    d = decide(ff, dcfg, ffaa_forgery_type=r.get("forgery_type"))
    result, forgery_type, match = d["decision"], d["forgery_type"], d["match_score"]
    face_liveness = {
        "Analysis result": result,
        "Forgery type": FORGERY_LABEL.get(forgery_type, "Fake") if result != "real" else "None",
        "Match score": f"{match:.4f}",
        "Forgery score": f"{ff:.4f}",
        "Model": f"PAAS_ensemble_v2 [{cfg.fusion.method}]",
        "Threshold": round(tau, 4),
    }
    return {
        "success": True,
        "decision": result,
        "forgery_type": forgery_type,
        "match_score": round(match, 4),
        "forgery_score": round(ff, 4),
        "processing_time_sec": r.get("processing_time_sec"),
        "face_liveness": face_liveness,
        "details": {
            "fusion": cfg.fusion.method,
            "ensemble_fake": r.get("ensemble_fake"),
            "ffaa_fake": r.get("ffaa_fake"),
            "ffaa_analysis": r.get("ffaa_analysis"),
            "threshold": round(tau, 4),
        },
    }


# --------------------------------------------------------------------------------------
# image I/O helpers
# --------------------------------------------------------------------------------------
def validate_suffix(image_bytes: bytes) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large")
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            im.load()
            fmt = (im.format or "").upper()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Failed to decode image") from exc
    return IMAGE_FORMAT_TO_EXT.get(fmt, ".png")


def client_ip(request: Optional[Request]) -> str:
    if request is None:
        return "unknown"
    ip = request.headers.get("X-Forwarded-For") or (request.client.host if request.client else None)
    return (ip.split(",")[0].strip() if ip else None) or "unknown"


def write_image(image_bytes: bytes, suffix: str, request: Optional[Request]) -> str:
    if SAVE_REQUESTS:
        subdir = os.path.join(REQUEST_DIR, client_ip(request))
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix)
        with open(path, "wb") as fh:
            fh.write(image_bytes)
        return path
    fd, path = tempfile.mkstemp(suffix=suffix, dir=REQUEST_DIR)
    with os.fdopen(fd, "wb") as fh:
        fh.write(image_bytes)
    return path


def cleanup(paths: List[str]) -> None:
    if SAVE_REQUESTS:
        return
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def decode_b64(image_base64: str) -> bytes:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(
    title="PAAS_ensemble_v2 Face Real/Fake API",
    version=os.environ.get("API_VERSION", "2.0-paas"),
    description="Combined FFAA (MLLM+MIDS) + 9-class MLLM-free ensemble, score-fused. /docs for Swagger.",
)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def add_headers(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Process-Time-Ms"] = f"{(time.time() - started) * 1000:.2f}"
    return response


class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 image (data URL prefix accepted).")
    threshold: Optional[float] = Field(None, description="Override decision threshold.")


class Base64BatchRequest(BaseModel):
    images_base64: Optional[List[str]] = Field(None, description="List of base64 images.")
    image_base64_list: Optional[List[str]] = Field(None, description="Backward-compatible alias.")
    threshold: Optional[float] = Field(None, description="Override decision threshold for all images.")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage():
    return ("<html><body><h3>PAAS_ensemble_v2 Face Real/Fake API</h3>"
            f"<p>fusion={cfg.fusion.method}, ffaa={engine.ffaa is not None}, ensemble9={engine.ens is not None}</p>"
            "<ul><li><a href='/docs'>Swagger UI</a></li><li><a href='/status'>Status</a></li></ul></body></html>")


@app.get("/health", tags=["status"])
async def health():
    return {"success": True, "status": "ok"}


@app.get("/status", tags=["status"])
async def status():
    return {
        "success": True,
        "config": cfg.name,
        "fusion": cfg.fusion.method,
        "ensemble_weight": cfg.fusion.ensemble_weight,
        "models": {"ffaa": engine.ffaa is not None, "ensemble9": engine.ens is not None},
        "device": cfg.device,
        "default_threshold": cfg.decision.threshold,
        "real_ambiguous_match_min": cfg.decision.real_ambiguous_match_min,
        "max_batch_size": MAX_BATCH_SIZE,
    }


@app.post("/face_liveness", tags=["liveness"], summary="Analyze an uploaded face image")
async def face_liveness(request: Request, face: UploadFile = File(...),
                        threshold: Optional[float] = Query(None)):
    if not face.filename:
        raise HTTPException(status_code=400, detail="no face image file.")
    image_bytes = await face.read()
    try:
        suffix = validate_suffix(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    path = write_image(image_bytes, suffix, request)
    t0 = time.perf_counter()
    try:
        res = await run_in_threadpool(score_paths, [path])
    finally:
        cleanup([path])
    r = res[0]
    r["processing_time_sec"] = round(time.perf_counter() - t0, 4)
    if r.get("decision") == "error":
        return JSONResponse({"success": False, "error": r.get("error", "inference failed")})
    return JSONResponse(build_response(r, threshold))


@app.post("/face_liveness_base64", tags=["liveness"], summary="Analyze a base64 face image")
async def face_liveness_base64(request: Request, payload: Base64ImageRequest = Body(...)):
    try:
        image_bytes = decode_b64(payload.image_base64)
        suffix = validate_suffix(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    path = write_image(image_bytes, suffix, request)
    t0 = time.perf_counter()
    try:
        res = await run_in_threadpool(score_paths, [path])
    finally:
        cleanup([path])
    r = res[0]
    r["processing_time_sec"] = round(time.perf_counter() - t0, 4)
    if r.get("decision") == "error":
        return JSONResponse({"success": False, "error": r.get("error", "inference failed")})
    return JSONResponse(build_response(r, payload.threshold))


@app.post("/face_liveness_base64_batch", tags=["liveness"], summary="Analyze a batch of base64 images")
async def face_liveness_base64_batch(request: Request, payload: Base64BatchRequest = Body(...)):
    images_base64 = payload.images_base64 or payload.image_base64_list
    if not isinstance(images_base64, list) or not images_base64:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list")
    if len(images_base64) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit of {MAX_BATCH_SIZE}")
    results: List[Optional[dict]] = [None] * len(images_base64)
    valid_idx, paths = [], []
    for i, b64 in enumerate(images_base64):
        if not isinstance(b64, str) or not b64.strip():
            results[i] = {"success": False, "error": "image_base64 must be a non-empty string"}
            continue
        try:
            image_bytes = decode_b64(b64)
            suffix = validate_suffix(image_bytes)
        except ValueError as exc:
            results[i] = {"success": False, "error": str(exc)}
            continue
        paths.append(write_image(image_bytes, suffix, request))
        valid_idx.append(i)
    if paths:
        t0 = time.perf_counter()
        try:
            res = await run_in_threadpool(score_paths, paths)
        finally:
            cleanup(paths)
        per_item = round((time.perf_counter() - t0) / max(len(paths), 1), 4)  # avg over the batch
        for local, original in enumerate(valid_idx):
            r = res[local]
            r["processing_time_sec"] = per_item
            results[original] = ({"success": False, "error": r.get("error", "inference failed")}
                                 if r.get("decision") == "error" else build_response(r, payload.threshold))
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"success": False, "error": "Unknown input error"}
    return JSONResponse({"success": True, "results": results})


if __name__ == "__main__":
    import uvicorn
    # pass the constructed app (not "module:app") so the models load once, not again in a re-import.
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "3000")), workers=1)
