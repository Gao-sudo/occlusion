"""Independent FastAPI service for occlusion-aware counting.

Runs on a separate port from api/app.py (default 8001) so both services can
coexist without any code changes to the existing API.

Start:
    uvicorn occlusion.api:app --host 0.0.0.0 --port 8001

Endpoints:
    POST /api/v1/occlusion/count
    POST /api/v1/occlusion/analyze
"""
from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from occlusion.config import (
    DEFAULT_CONF,
    DEFAULT_DATA_YAML,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_PRETRAINED_SEG,
    PROJECT_ROOT,
)
from occlusion.depth_estimator import DepthEstimator
from occlusion.pipeline import process_image
from occlusion.utils import load_class_names

app = FastAPI(title="Jomoo Occlusion Counting API", version="1.1.0")


class OcclusionSettings:
    def __init__(self) -> None:
        self.device = os.environ.get("OCCLUSION_DEVICE", DEFAULT_DEVICE)
        self.imgsz = int(os.environ.get("OCCLUSION_IMGSZ", str(DEFAULT_IMGSZ)))
        self.conf = float(os.environ.get("OCCLUSION_CONF", str(DEFAULT_CONF)))
        self.iou = float(os.environ.get("OCCLUSION_IOU", str(DEFAULT_IOU)))
        self.max_det = int(os.environ.get("OCCLUSION_MAX_DET", str(DEFAULT_MAX_DET)))
        self.weights = os.environ.get("OCCLUSION_WEIGHTS", None)
        self.data_yaml = Path(os.environ.get("OCCLUSION_DATA_YAML", str(DEFAULT_DATA_YAML)))
        self.depth_encoder = os.environ.get("OCCLUSION_DEPTH_ENCODER", "vitb")
        self.depth_weights = os.environ.get("OCCLUSION_DEPTH_WEIGHTS", None)
        self.skip_depth = os.environ.get("OCCLUSION_SKIP_DEPTH", "false").lower() == "true"


@lru_cache(maxsize=1)
def get_settings() -> OcclusionSettings:
    return OcclusionSettings()


@lru_cache(maxsize=1)
def get_seg_model() -> YOLO:
    settings = get_settings()
    weights = settings.weights
    if weights is None:
        candidates = sorted((PROJECT_ROOT / "outputs" / "occlusion").rglob("best.pt")) if (PROJECT_ROOT / "outputs" / "occlusion").exists() else []
        if candidates:
            weights = str(candidates[-1])
        else:
            weights = str(DEFAULT_PRETRAINED_SEG)
    return YOLO(weights)


@lru_cache(maxsize=1)
def get_depth_estimator() -> DepthEstimator | None:
    settings = get_settings()
    if settings.skip_depth:
        return None
    try:
        return DepthEstimator(
            encoder=settings.depth_encoder,
            device=settings.device,
            weights_path=settings.depth_weights,
        )
    except Exception as exc:
        print(f"[WARNING] Depth estimator failed to load: {exc}")
        return None


@lru_cache(maxsize=1)
def get_class_names() -> list[str]:
    settings = get_settings()
    return load_class_names(settings.data_yaml)


@app.exception_handler(Exception)
async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": 1004, "msg": "Server internal error", "data": None},
    )


def _decode_image(payload: bytes) -> np.ndarray | None:
    if not payload:
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _is_allowed_file(upload: UploadFile) -> bool:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png"}


def _process_image(image_bgr: np.ndarray) -> dict[str, Any]:
    settings = get_settings()
    seg_model = get_seg_model()
    depth_estimator = get_depth_estimator()
    class_names = get_class_names()

    out = process_image(
        image_bgr=image_bgr,
        seg_model=seg_model,
        depth_estimator=depth_estimator,
        class_names=class_names,
        imgsz=settings.imgsz,
        conf=settings.conf,
        iou=settings.iou,
        max_det=settings.max_det,
        device=settings.device,
    )

    # Encode visualization to base64 for optional return
    vis_image = out["vis_image"]
    _, encoded = cv2.imencode(".jpg", vis_image)
    vis_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8") if encoded is not None else ""
    out["visualization_base64"] = vis_b64
    return out


@app.post("/api/v1/occlusion/count")
@app.post("/api/occlusion/count")
async def count_endpoint(images: list[UploadFile] = File(...)) -> Any:
    if not images:
        return JSONResponse(status_code=400, content={"code": 1001, "msg": "No image uploaded", "data": None})
    if any(not _is_allowed_file(u) for u in images):
        return JSONResponse(status_code=400, content={"code": 1002, "msg": "Invalid file format", "data": None})

    payload_results: list[dict[str, Any]] = []
    for upload in images:
        data = await upload.read()
        image = _decode_image(data)
        if image is None:
            return JSONResponse(status_code=400, content={"code": 1002, "msg": "Invalid image data", "data": None})
        out = _process_image(image)
        payload_results.append({
            "filename": upload.filename,
            "summary": out["summary"],
        })

    return {
        "code": 200,
        "msg": "success",
        "data": payload_results,
    }


@app.post("/api/v1/occlusion/analyze")
@app.post("/api/occlusion/analyze")
async def analyze_endpoint(images: list[UploadFile] = File(...)) -> Any:
    if not images:
        return JSONResponse(status_code=400, content={"code": 1001, "msg": "No image uploaded", "data": None})
    if any(not _is_allowed_file(u) for u in images):
        return JSONResponse(status_code=400, content={"code": 1002, "msg": "Invalid file format", "data": None})

    payload_results: list[dict[str, Any]] = []
    for upload in images:
        data = await upload.read()
        image = _decode_image(data)
        if image is None:
            return JSONResponse(status_code=400, content={"code": 1002, "msg": "Invalid image data", "data": None})
        out = _process_image(image)
        payload_results.append({
            "filename": upload.filename,
            "summary": out["summary"],
            "instances": out["instances"],
            "filtered_instances": out["filtered_instances"],
            "visualization_base64": out["visualization_base64"],
        })

    return {
        "code": 200,
        "msg": "success",
        "data": payload_results,
    }
