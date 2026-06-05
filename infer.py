"""Inference pipeline for occlusion-aware counting.

Example:
    python -m occlusion.infer \
        --source ./test_imgs \
        --weights ./outputs/yolov11s_seg/best.pt \
        --device 0
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from occlusion.config import (
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRETRAINED_SEG,
    PROJECT_ROOT,
)
from occlusion.depth_estimator import DepthEstimator
from occlusion.fusion_counter import count_all_clusters, summarize_counts
from occlusion.mask_analyzer import cluster_masks, extract_mask_infos
from occlusion.utils import (
    build_data_yaml_seg,
    ensure_dir,
    load_class_names,
    load_image,
    save_image,
    save_json,
)
from occlusion.visualizer import compose_result_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occlusion-aware counting inference")
    parser.add_argument("--source", type=Path, required=True, help="Directory or single image path")
    parser.add_argument("--weights", type=str, default=None, help="Path to YOLO-seg weights")
    parser.add_argument("--data-yaml", type=Path, default=PROJECT_ROOT / "data.yaml")
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
    image_bgr = load_image(image_path)

    # --- Segmentation ---
    results = seg_model.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        max_det=max_det,
        verbose=False,
    )
    result = results[0]

    masks_np = result.masks.data.cpu().numpy() if result.masks is not None else np.array([])
    if masks_np.ndim == 3:
        # Ultralytics may return masks in original image resolution or model resolution
        # Ensure binary masks resized to original image size
        h, w = image_bgr.shape[:2]
        resized_masks = []
        for m in masks_np:
            rm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(rm > 0)
        masks_np = np.stack(resized_masks) if resized_masks else np.array([])

    class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None and result.boxes.cls is not None else np.array([])
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else None

    mask_infos = extract_mask_infos(masks_np, class_ids, class_names, confidences)
    clusters = cluster_masks(mask_infos)

    # --- Depth estimation ---
    depth_map = None
    if depth_estimator is not None:
        depth_map = depth_estimator.infer(image_bgr)
    else:
        # Create a dummy depth map for mask-only mode
        depth_map = np.zeros(image_bgr.shape[:2], dtype=np.float32)

    # --- Fusion counting ---
    count_results = count_all_clusters(clusters, depth_map)
    summary = summarize_counts(count_results)
    summary["image"] = str(image_path)

    # --- Visualization ---
    vis_image = compose_result_image(image_bgr, depth_map, clusters, count_results)

    return {
        "summary": summary,
        "vis_image": vis_image,
        "clusters": clusters,
        "count_results": count_results,
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    images = _list_images(source)
    if not images:
        raise RuntimeError(f"No images found in {source}")

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(args.output_root / "occlusion_infer" / tag)
    vis_dir = ensure_dir(out_dir / "visualizations")
    meta_dir = ensure_dir(out_dir / "meta")

    class_names = load_class_names(args.data_yaml)

    # Load segmentation model
    weights_path = args.weights
    if weights_path is None:
        # Try to find latest trained seg weight or fallback to pretrained
        candidates = sorted((PROJECT_ROOT / "outputs" / "occlusion").rglob("best.pt")) if (PROJECT_ROOT / "outputs" / "occlusion").exists() else []
        if candidates:
            weights_path = str(candidates[-1])
        else:
            weights_path = str(DEFAULT_PRETRAINED_SEG)
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
        all_summaries.append(out["summary"])
        save_image(vis_dir / img_path.name, out["vis_image"])

    # Save aggregated results
    save_json(meta_dir / "results.json", all_summaries)
    print(f"[Done] Results saved to {out_dir}")
    print(f"       Visualizations: {vis_dir}")
    print(f"       JSON meta: {meta_dir / 'results.json'}")


if __name__ == "__main__":
    main()
