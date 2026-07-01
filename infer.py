"""Inference pipeline for occlusion-aware counting.

Example:
    python -m occlusion.infer \
        --source ./test_imgs \
        --weights ./outputs/yolov11s_seg/best.pt \
        --device 0
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRETRAINED_SEG,
    PROJECT_ROOT,
)
from occlusion.depth_estimator import DepthEstimator
from occlusion.pipeline import process_image
from occlusion.utils import ensure_dir, load_class_names, save_image, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occlusion-aware counting inference")
    parser.add_argument("--source", type=Path, required=True, help="Directory or single image path")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLO-seg weights")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--depth-encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--depth-weights", type=str, default=None, help="Path to DA-V2 weights")
    parser.add_argument("--skip-depth", action="store_true", help="Skip depth estimation; use mask-only heuristic")
    return parser.parse_args()


def _resolve_input_path(path: Path) -> Path:
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                PROJECT_ROOT / path,
                PROJECT_ROOT.parent / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else path.resolve()


def _list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def run_single_image(
    image_path: Path,
    seg_model: YOLO,
    depth_estimator: DepthEstimator | None,
    class_names: list[str],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> dict[str, Any]:
    """Run occlusion counting on a single image."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    out = process_image(
        image_bgr=image_bgr,
        seg_model=seg_model,
        depth_estimator=depth_estimator,
        class_names=class_names,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=str(device),
    )
    out["summary"]["image"] = str(image_path)
    return out


def main() -> None:
    args = parse_args()
    source = _resolve_input_path(args.source)
    images = _list_images(source)
    if not images:
        raise RuntimeError(f"No images found in {source}")

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(args.output_root / "occlusion_infer" / tag)
    vis_dir = ensure_dir(out_dir / "visualizations")
    meta_dir = ensure_dir(out_dir / "meta")

    class_names = load_class_names(_resolve_input_path(args.data_yaml))

    # Load segmentation model
    weights_path = args.weights
    if weights_path is None:
        candidates = sorted((PROJECT_ROOT / "outputs" / "occlusion").rglob("best.pt")) if (PROJECT_ROOT / "outputs" / "occlusion").exists() else []
        if candidates:
            weights_path = str(candidates[-1])
        else:
            weights_path = str(DEFAULT_PRETRAINED_SEG)
    else:
        weights_path = str(_resolve_input_path(Path(weights_path)))
    print(f"[Seg model] Loading: {weights_path}")
    seg_model = YOLO(weights_path)

    # Load depth estimator
    depth_estimator = None
    if not args.skip_depth:
        print(f"[Depth model] Loading DA-V2 encoder={args.depth_encoder}")
        try:
            depth_estimator = DepthEstimator(
                encoder=args.depth_encoder,
                device=str(args.device),
                weights_path=args.depth_weights,
            )
        except Exception as exc:
            print(f"[WARNING] Failed to load depth estimator: {exc}")
            print("[WARNING] Falling back to mask-only heuristic counting.")

    all_summaries = []
    for img_path in images:
        print(f"[Infer] {img_path.name}")
        out = run_single_image(
            img_path,
            seg_model,
            depth_estimator,
            class_names,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=str(args.device),
        )
        record = {
            "summary": out["summary"],
            "instances": out["instances"],
            "filtered_instances": out["filtered_instances"],
        }
        all_summaries.append(record)
        save_image(vis_dir / img_path.name, out["vis_image"])

    # Save aggregated results
    save_json(meta_dir / "results.json", all_summaries)
    print(f"[Done] Results saved to {meta_dir / 'results.json'}")


if __name__ == "__main__":
    main()
