"""Shared inference pipeline for occlusion-aware counting.

This module consolidates the duplicated logic previously present in
`occlusion/infer.py` and `occlusion/api.py`. It runs the full
segmentation -> mask analysis -> context-aware decision -> counting
pipeline and returns a structured result dictionary.
"""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from occlusion.config import DEFAULT_DATA_YAML
from occlusion.decision_engine import classify_all_instance_decisions
from occlusion.depth_estimator import DepthEstimator
from occlusion.fusion_counter import count_all_clusters, estimate_unit_depth_per_class, summarize_counts
from occlusion.label_convert import build_detection_polygon, polygon_to_points_list
from occlusion.mask_analyzer import (
    ClusterInfo,
    MaskInfo,
    build_cluster_by_source_index,
    classify_clusters_countability,
    cluster_masks,
    extract_mask_infos,
    filter_top_horizontal_display_masks,
    resolve_cluster_dominant_sku,
)
from occlusion.utils import load_class_names
from occlusion.visualizer import compose_result_image


def _serialize_instance(
    mask_info: MaskInfo,
    masks_np: np.ndarray,
    image_shape: tuple[int, int],
    cluster: ClusterInfo | None = None,
    countability: str = "countable",
    countability_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a MaskInfo into the output JSON dict."""
    source_index = mask_info.source_index
    mask = masks_np[source_index] if source_index < len(masks_np) else None
    polygon, polygon_source = build_detection_polygon(
        mask=mask,
        bbox_xyxy=(mask_info.x1, mask_info.y1, mask_info.x2, mask_info.y2),
        image_shape=image_shape,
    )

    # SKU inheritance: use cluster-dominant SKU for occluded/low-conf items.
    if cluster is not None:
        sku_id = cluster.dominant_class_id if cluster.dominant_class_id is not None else mask_info.class_id
        sku_name = cluster.dominant_class_name if cluster.dominant_class_name is not None else mask_info.class_name
        sku_source = (
            "direct_detection"
            if mask_info.source_index in cluster.high_conf_source_indices
            else "cluster_inheritance"
        )
    else:
        sku_id = mask_info.class_id
        sku_name = mask_info.class_name
        sku_source = "direct_detection"

    return {
        "class_id": mask_info.class_id,
        "class_name": mask_info.class_name,
        "confidence": mask_info.confidence,
        "bbox": [mask_info.x1, mask_info.y1, mask_info.x2, mask_info.y2],
        "polygon": polygon_to_points_list(polygon),
        "polygon_source": polygon_source,
        "area_px": mask_info.area_px,
        "centroid": [mask_info.cx, mask_info.cy],
        "orientation_deg": mask_info.orientation_deg,
        "decision": mask_info.decision,
        "decision_reasons": list(mask_info.decision_reasons),
        "sku_id": sku_id,
        "sku_name": sku_name,
        "sku_source": sku_source,
        "countability": countability,
        "countability_reasons": list(countability_reasons or []),
    }


def process_image(
    image_bgr: np.ndarray,
    seg_model: YOLO,
    depth_estimator: DepthEstimator | None,
    class_names: list[str],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    display_roi: tuple[int, int, int, int] | Sequence[int] | None = None,
    data_yaml: Any = None,
) -> dict[str, Any]:
    """Run the full occlusion counting pipeline on a single image.

    Args:
        image_bgr: input image in BGR format.
        seg_model: loaded YOLO-seg model.
        depth_estimator: optional DepthEstimator (pass None to skip depth).
        class_names: list of class names from data.yaml.
        imgsz, conf, iou, max_det, device: YOLO inference parameters.
        display_roi: optional bounding box (x1, y1, x2, y2) of the display area.
        data_yaml: unused, kept for backward-compatible call signatures.

    Returns:
        Dictionary with keys:
            summary, instances, filtered_instances, vis_image,
            clusters, count_results.
    """
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
        h, w = image_bgr.shape[:2]
        resized_masks = []
        for m in masks_np:
            rm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
            resized_masks.append(rm > 0)
        masks_np = np.stack(resized_masks) if resized_masks else np.array([])

    class_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None and result.boxes.cls is not None else np.array([])
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else None

    image_shape = image_bgr.shape[:2]

    # --- Mask geometry analysis ---
    mask_infos = extract_mask_infos(masks_np, class_ids, class_names, confidences)
    mask_infos, filtered_mask_infos = filter_top_horizontal_display_masks(mask_infos, image_shape)
    clusters = cluster_masks(mask_infos)
    cluster_by_source_index = build_cluster_by_source_index(clusters)

    # --- Cluster SKU inheritance ---
    for cluster in clusters:
        resolve_cluster_dominant_sku(cluster)

    # --- Context-aware instance decisions ---
    classify_all_instance_decisions(mask_infos, cluster_by_source_index, image_shape, display_roi=display_roi)

    # --- Serialize instances (before countability classification, defaults to countable) ---
    instances = []
    for mask_info in mask_infos:
        cluster = cluster_by_source_index.get(mask_info.source_index)
        instances.append(
            _serialize_instance(
                mask_info,
                masks_np,
                image_shape,
                cluster=cluster,
                countability=cluster.countability if cluster else "countable",
                countability_reasons=cluster.countability_reasons if cluster else [],
            )
        )

    filtered_instances = [
        {
            **_serialize_instance(mask_info, masks_np, image_shape),
            "decision": "filtered",
            "decision_reasons": ["top_horizontal_display_sign"],
            "filter_reason": "top_horizontal_display_sign",
        }
        for mask_info in filtered_mask_infos
    ]

    # --- Depth estimation (optional) ---
    depth_map = None
    if depth_estimator is not None:
        depth_map = depth_estimator.infer(image_bgr)
    else:
        depth_map = np.zeros(image_bgr.shape[:2], dtype=np.float32)

    # --- Countability classification ---
    clusters = classify_clusters_countability(clusters, depth_map)

    # --- Learn unit depth per item from countable clusters ---
    unit_depth_map = estimate_unit_depth_per_class(clusters, depth_map)

    # --- Fusion counting ---
    count_results = count_all_clusters(clusters, depth_map, unit_depth_map=unit_depth_map)
    summary = summarize_counts(count_results, filtered_count=len(filtered_mask_infos))

    # --- Visualization ---
    vis_image = compose_result_image(image_bgr, depth_map, clusters, count_results)

    return {
        "summary": summary,
        "instances": instances,
        "filtered_instances": filtered_instances,
        "vis_image": vis_image,
        "clusters": clusters,
        "count_results": count_results,
    }


def load_class_names_for_pipeline(data_yaml: Any = None) -> list[str]:
    """Load class names, defaulting to the project data.yaml."""
    if data_yaml is None:
        data_yaml = DEFAULT_DATA_YAML
    return load_class_names(data_yaml)
