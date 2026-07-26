"""FastAPI service for product visible-count inference.

Start:
    uvicorn api:app --host 0.0.0.0 --port 8001

Endpoint:
    POST /api/v1/count/batch
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from occlusion.utils import ensure_dir, load_class_names, save_image

app = FastAPI(title="Product Visible Count API", version="1.0.0")

DEFAULT_BEST_WEIGHTS = PROJECT_ROOT / "best.pt"
API_VIS_ROOT = ensure_dir(PROJECT_ROOT / "outputs" / "api_visualizations")


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
        if DEFAULT_BEST_WEIGHTS.exists():
            weights = str(DEFAULT_BEST_WEIGHTS)
        else:
            candidates = sorted((PROJECT_ROOT / "outputs" / "occlusion").rglob("best.pt")) if (PROJECT_ROOT / "outputs" / "occlusion").exists() else []
            if candidates:
                weights = str(candidates[-1])
            else:
                weights = str(DEFAULT_PRETRAINED_SEG)
    return YOLO(weights)


@lru_cache(maxsize=1)
def get_depth_estimator() -> Optional[DepthEstimator]:
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


def _decode_image(payload: bytes) -> Optional[np.ndarray]:
    if not payload:
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _is_allowed_file(upload: UploadFile) -> bool:
    suffix = Path(upload.filename or "").suffix.lower()
    return suffix in {".jpg", ".jpeg", ".png"}


def _process_image(image_bgr: np.ndarray, include_visualization: bool = False) -> dict[str, Any]:
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
        data_yaml=settings.data_yaml,
    )

    if include_visualization:
        vis_image = out["vis_image"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        vis_dir = ensure_dir(API_VIS_ROOT / timestamp[:8])
        vis_name = f"vis_{timestamp}.jpg"
        vis_path = vis_dir / vis_name
        save_image(vis_path, vis_image)
        out["visualization_path"] = str(vis_path)
    return out


def _compact_result(filename: str | None, out: dict[str, Any], include_instances: bool, include_visualization: bool) -> dict[str, Any]:
    summary = out["summary"]
    per_class_summary = summary.get("per_class_summary", {})
    if isinstance(per_class_summary, dict):
        items = [
            {
                "category": str(category),
                "count": int(values.get("total", 0)) if isinstance(values, dict) else int(values),
            }
            for category, values in per_class_summary.items()
        ]
    else:
        items = []

    result: dict[str, Any] = {
        "filename": filename,
        "total_count": int(summary.get("total_visible", 0)),
        "items": items,
    }
    if include_instances:
        result["instances"] = out["instances"]
        result["filtered_instances"] = out["filtered_instances"]
    if include_visualization:
        result["visualization_path"] = out.get("visualization_path", "")
    return result


@app.post("/api/v1/count/batch")
async def batch_count_endpoint(
    images: list[UploadFile] = File(...),
    include_instances: bool = Form(False),
    include_visualization: bool = Form(False),
) -> Any:
    """Batch visible-count inference.

    Request:
        multipart/form-data
        - images: one or more jpg/jpeg/png files
        - include_instances: optional bool, default false
        - include_visualization: optional bool, default false
    """
    if not images:
        return JSONResponse(status_code=400, content={"code": 1001, "msg": "No image uploaded", "data": None})
    if any(not _is_allowed_file(u) for u in images):
        return JSONResponse(status_code=400, content={"code": 1002, "msg": "Only jpg, jpeg and png files are supported", "data": None})

    results: list[dict[str, Any]] = []
    for upload in images:
        data = await upload.read()
        image = _decode_image(data)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"code": 1002, "msg": f"Invalid image data: {upload.filename}", "data": None},
            )
        out = _process_image(image, include_visualization=include_visualization)
        results.append(_compact_result(upload.filename, out, include_instances, include_visualization))

    return {
        "code": 200,
        "msg": "success",
        "data": {
            "total_images": len(results),
            "results": results,
        },
    }
