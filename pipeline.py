"""Shared inference pipeline for occlusion-aware counting.

This module consolidates the duplicated logic previously present in
`occlusion/infer.py` and `occlusion/api.py`. It runs the full
segmentation -> mask analysis -> context-aware decision -> counting
pipeline and returns a structured result dictionary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from occlusion.config import CLUSTER_EPS_PX, DEFAULT_DATA_YAML
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
    filter_instances_by_class_priors,
    filter_counting_instances,
    filter_top_horizontal_display_masks,
    is_regular_dense_display_cluster,
    is_regular_dense_overcount_cluster,
    load_class_priors,
    merge_physical_item_fragments,
    resolve_cluster_dominant_sku,
)
from occlusion.utils import load_class_names
from occlusion.visualizer import compose_result_image


def _serialize_instance(
    mask_info: MaskInfo,
    masks_np: np.ndarray,
    image_shape: tuple[int, int],
    cluster: Optional[ClusterInfo] = None,
    countability: str = "countable",
    countability_reasons: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Convert a MaskInfo into the output JSON dict."""
    source_index = mask_info.source_index
    mask = masks_np[source_index] if source_index < len(masks_np) else mask_info.mask
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


def _build_local_dense_candidate_clusters(
    mask_infos: list[MaskInfo],
    base_clusters: list[ClusterInfo],
    eps_px: float,
) -> list[ClusterInfo]:
    """Build local dense groups from centroid proximity for region-level reasoning."""
    if len(mask_infos) < 2:
        return []
    if not any(cluster.countability == "uncountable" for cluster in base_clusters):
        return []

    points = np.asarray([[m.cx, m.cy] for m in mask_infos], dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    nearest = np.partition(distances, 1, axis=1)[:, 1]
    dense_indices = np.where(nearest <= eps_px)[0]
    if len(dense_indices) < 2:
        return []

    local_masks = [mask_infos[int(index)] for index in dense_indices]
    return cluster_masks(local_masks, eps_px=eps_px)


def _refine_dense_cluster_rois(
    image_bgr: np.ndarray,
    seg_model: YOLO,
    mask_infos: list[MaskInfo],
    masks_np: np.ndarray,
    clusters: list[ClusterInfo],
    class_names: list[str],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> tuple[list[MaskInfo], np.ndarray, int, int, list[tuple[int, int, int, int]]]:
    """Re-run dense cluster ROIs and append only spatially new mask evidence."""
    image_h, image_w = image_bgr.shape[:2]
    next_source_index = len(masks_np)
    roi_count = 0
    roi_boxes: list[tuple[int, int, int, int]] = []
    added_infos: list[MaskInfo] = []
    added_masks: list[np.ndarray] = []

    for cluster in clusters:
        if len(cluster.masks) < 2:
            continue
        x1 = max(0, min(m.x1 for m in cluster.masks))
        y1 = max(0, min(m.y1 for m in cluster.masks))
        x2 = min(image_w, max(m.x2 for m in cluster.masks) + 1)
        y2 = min(image_h, max(m.y2 for m in cluster.masks) + 1)
        pad_x = max(8, int(round((x2 - x1) * 0.10)))
        pad_y = max(8, int(round((y2 - y1) * 0.10)))
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(image_w, x2 + pad_x)
        ry2 = min(image_h, y2 + pad_y)
        roi = image_bgr[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        roi_count += 1
        roi_boxes.append((rx1, ry1, rx2, ry2))

        roi_result = seg_model.predict(
            source=roi,
            imgsz=max(imgsz, 896),
            conf=max(0.10, conf * 0.60),
            iou=max(0.80, iou),
            device=device,
            max_det=max_det,
            verbose=False,
        )[0]
        if roi_result.masks is None or roi_result.boxes is None:
            continue

        local_masks = roi_result.masks.data.cpu().numpy()
        local_classes = roi_result.boxes.cls.cpu().numpy() if roi_result.boxes.cls is not None else np.array([])
        local_confidences = roi_result.boxes.conf.cpu().numpy() if roi_result.boxes.conf is not None else None
        if local_masks.ndim != 3:
            continue

        roi_h, roi_w = roi.shape[:2]
        cluster_source_ids = {m.source_index for m in cluster.masks}
        removed_source_ids: set[int] = set()
        existing_masks = [m.mask for m in cluster.masks]
        for local_index, local_mask in enumerate(local_masks):
            resized = cv2.resize(local_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR) > 0
            global_mask = np.zeros((image_h, image_w), dtype=bool)
            global_mask[ry1:ry2, rx1:rx2] = resized
            if not global_mask.any():
                continue

            candidate_info = extract_mask_infos(
                np.asarray([global_mask]),
                np.asarray([int(local_classes[local_index]) if local_index < len(local_classes) else 0]),
                class_names,
                np.asarray([
                    float(local_confidences[local_index])
                    if local_confidences is not None and local_index < len(local_confidences)
                    else 1.0
                ]),
            )
            if not candidate_info:
                continue
            candidate = candidate_info[0]

            matched_infos: list[MaskInfo] = []
            overlaps_existing = False
            near_existing = False
            for existing_info in cluster.masks:
                if existing_info.source_index in removed_source_ids:
                    continue
                intersection = np.logical_and(global_mask, existing_info.mask).sum()
                union = np.logical_or(global_mask, existing_info.mask).sum()
                overlap = float(intersection) / max(1, int(union))
                center_distance = float(np.hypot(candidate.cx - existing_info.cx, candidate.cy - existing_info.cy))
                candidate_size = max(candidate.x2 - candidate.x1, candidate.y2 - candidate.y1, 1)
                existing_size = max(existing_info.x2 - existing_info.x1, existing_info.y2 - existing_info.y1, 1)
                if overlap >= 0.20:
                    overlaps_existing = True
                    matched_infos.append(existing_info)
                if center_distance < 0.55 * max(candidate_size, existing_size):
                    near_existing = True

            class_id = int(local_classes[local_index]) if local_index < len(local_classes) else 0
            if class_id < 0 or class_id >= len(class_names):
                continue
            confidence = (
                float(local_confidences[local_index])
                if local_confidences is not None and local_index < len(local_confidences)
                else 1.0
            )
            refined_infos = extract_mask_infos(
                np.asarray([global_mask]),
                np.asarray([class_id]),
                class_names,
                np.asarray([confidence]),
            )
            if not refined_infos:
                continue
            refined_info = refined_infos[0]

            if len(matched_infos) >= 2:
                candidate_area = refined_info.area_px
                matched_area = max(m.area_px for m in matched_infos)
                if candidate_area >= 0.80 * matched_area:
                    for matched in matched_infos:
                        removed_source_ids.add(matched.source_index)
                    refined_info.source_index = next_source_index
                    refined_info.decision_reasons = ["dense_roi_refine_replace"]
                    added_infos.append(refined_info)
                    added_masks.append(global_mask)
                    existing_masks.append(global_mask)
                    next_source_index += 1
                    continue

            if overlaps_existing or near_existing:
                continue

            refined_info.source_index = next_source_index
            refined_info.decision_reasons = ["dense_roi_refine_candidate"]
            added_infos.append(refined_info)
            added_masks.append(global_mask)
            existing_masks.append(global_mask)
            next_source_index += 1

        if removed_source_ids:
            mask_infos = [m for m in mask_infos if m.source_index not in removed_source_ids]

    if not added_infos:
        return mask_infos, masks_np, 0, roi_count, roi_boxes
    return (
        mask_infos + added_infos,
        np.concatenate([masks_np, np.asarray(added_masks, dtype=bool)], axis=0),
        len(added_infos),
        roi_count,
        roi_boxes,
    )


def process_image(
    image_bgr: np.ndarray,
    seg_model: YOLO,
    depth_estimator: Optional[DepthEstimator],
    class_names: list[str],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    display_roi: Optional[Union[Tuple[int, int, int, int], Sequence[int]]] = None,
    data_yaml: Any = None,
    class_priors_path: Optional[Union[Path, str]] = None,
    enable_class_priors: bool = False,
    enable_roi_refine: bool = False,
    enable_physical_merge: bool = False,
) -> dict[str, Any]:
    """Run the full occlusion counting pipeline on a single image.

    Args:
        image_bgr: input image in BGR format.
        seg_model: loaded YOLO-seg model.
        depth_estimator: optional DepthEstimator (pass None to skip depth).
        class_names: list of class names from data.yaml.
        imgsz, conf, iou, max_det, device: YOLO inference parameters.
        display_roi: optional bounding box (x1, y1, x2, y2) of the display area.
        data_yaml: optional data.yaml path used to discover class_priors.json.
        class_priors_path: optional path to class geometry priors.
        enable_class_priors: enable class-level geometry prior filtering.
        enable_roi_refine: enable dense ROI second-pass segmentation.
        enable_physical_merge: merge strong same-item mask fragments before counting.

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
    priors_path = class_priors_path
    if priors_path is None and data_yaml is not None:
        try:
            priors_path = Path(data_yaml).parent / "class_priors.json"
        except TypeError:
            priors_path = None
    class_priors = load_class_priors(priors_path)

    # --- Mask geometry analysis ---
    mask_infos = extract_mask_infos(masks_np, class_ids, class_names, confidences)
    mask_infos, filtered_mask_infos = filter_top_horizontal_display_masks(mask_infos, image_shape)
    single_class_mode = len(class_names) == 1 and class_names[0] == "countable_product"
    mask_infos, prior_filtered_mask_infos = filter_instances_by_class_priors(
        mask_infos,
        image_shape,
        class_priors=class_priors if enable_class_priors and not single_class_mode else None,
    )
    cluster_eps_px = CLUSTER_EPS_PX * (2.0 if single_class_mode else 1.0)
    clusters = cluster_masks(mask_infos, eps_px=cluster_eps_px)
    cluster_by_source_index = build_cluster_by_source_index(clusters)

    # --- Cluster SKU inheritance ---
    for cluster in clusters:
        resolve_cluster_dominant_sku(cluster)

    # --- Context-aware instance decisions ---
    classify_all_instance_decisions(mask_infos, cluster_by_source_index, image_shape, display_roi=display_roi)

    # --- Early depth estimation / dense-scene classification for count-policy routing ---
    depth_map = None
    if depth_estimator is not None:
        depth_map = depth_estimator.infer(image_bgr)
    else:
        depth_map = np.zeros(image_bgr.shape[:2], dtype=np.float32)

    clusters = classify_clusters_countability(clusters, depth_map)
    cluster_by_source_index = build_cluster_by_source_index(clusters)
    local_dense_candidate_clusters = _build_local_dense_candidate_clusters(
        mask_infos,
        clusters,
        eps_px=cluster_eps_px * 3.0,
    )
    regular_dense_cluster_ids = [
        cluster.cluster_id
        for cluster in local_dense_candidate_clusters
        if is_regular_dense_display_cluster(cluster)
    ]
    regular_dense_overcount_cluster_ids = [
        cluster.cluster_id
        for cluster in local_dense_candidate_clusters
        if is_regular_dense_overcount_cluster(cluster)
    ]

    roi_refine_added = 0
    roi_refine_regions = 0
    roi_boxes: list[tuple[int, int, int, int]] = []
    if enable_roi_refine:
        # Dense ROI second pass: use local high-resolution segmentation to recover
        # spatially distinct evidence before applying counting filters.
        initial_mask_count = len(mask_infos)
        roi_candidate_clusters = _build_local_dense_candidate_clusters(
            mask_infos,
            clusters,
            eps_px=cluster_eps_px * 3.0,
        )
        mask_infos, masks_np, roi_refine_added, roi_refine_regions, roi_boxes = _refine_dense_cluster_rois(
            image_bgr=image_bgr,
            seg_model=seg_model,
            mask_infos=mask_infos,
            masks_np=masks_np,
            clusters=roi_candidate_clusters,
            class_names=class_names,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
        )
        if len(mask_infos) > initial_mask_count:
            clusters = cluster_masks(mask_infos, eps_px=cluster_eps_px)
            cluster_by_source_index = build_cluster_by_source_index(clusters)
            for cluster in clusters:
                resolve_cluster_dominant_sku(cluster)
            classify_all_instance_decisions(mask_infos, cluster_by_source_index, image_shape, display_roi=display_roi)
            clusters = classify_clusters_countability(clusters, depth_map)
            cluster_by_source_index = build_cluster_by_source_index(clusters)

    # Remove conservative duplicate/low-context-confidence detections before counting,
    # but keep more evidence inside dense/uncountable clusters.
    mask_infos, quality_filtered_mask_infos = filter_counting_instances(
        mask_infos,
        cluster_by_source_index=cluster_by_source_index,
    )
    fragment_filtered_mask_infos: list[MaskInfo] = []
    physical_item_groups: list[list[int]] = []
    if enable_physical_merge:
        mask_infos, fragment_filtered_mask_infos, physical_item_groups = merge_physical_item_fragments(mask_infos)
    clusters = cluster_masks(mask_infos, eps_px=cluster_eps_px)
    cluster_by_source_index = build_cluster_by_source_index(clusters)
    for cluster in clusters:
        resolve_cluster_dominant_sku(cluster)
    clusters = classify_clusters_countability(clusters, depth_map)
    cluster_by_source_index = build_cluster_by_source_index(clusters)

    filtered_instances = [
        {
            **_serialize_instance(mask_info, masks_np, image_shape),
            "decision": "filtered",
            "decision_reasons": ["top_horizontal_display_sign"],
            "filter_reason": "top_horizontal_display_sign",
        }
        for mask_info in filtered_mask_infos
    ]
    filtered_instances.extend(
        {
            **_serialize_instance(mask_info, masks_np, image_shape),
            "decision": "filtered",
            "decision_reasons": list(mask_info.decision_reasons),
            "filter_reason": mask_info.decision_reasons[-1] if mask_info.decision_reasons else "quality_filter",
        }
        for mask_info in quality_filtered_mask_infos
    )
    filtered_instances.extend(
        {
            **_serialize_instance(mask_info, masks_np, image_shape),
            "decision": "filtered",
            "decision_reasons": list(mask_info.decision_reasons),
            "filter_reason": mask_info.decision_reasons[-1] if mask_info.decision_reasons else "class_prior_filter",
        }
        for mask_info in prior_filtered_mask_infos
    )
    filtered_instances.extend(
        {
            **_serialize_instance(mask_info, masks_np, image_shape),
            "decision": "filtered",
            "decision_reasons": list(mask_info.decision_reasons),
            "filter_reason": "physical_item_fragment",
        }
        for mask_info in fragment_filtered_mask_infos
    )

    # --- Countability classification (final pass after count-input filtering) ---
    cluster_by_source_index = build_cluster_by_source_index(clusters)

    # --- Serialize instances with final cluster countability ---
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

    # --- Learn unit depth per item from countable clusters ---
    unit_depth_map = estimate_unit_depth_per_class(clusters, depth_map)

    # --- Fusion counting ---
    count_results = count_all_clusters(clusters, depth_map, unit_depth_map=unit_depth_map)
    summary = summarize_counts(
        count_results,
        filtered_count=(
            len(filtered_mask_infos)
            + len(prior_filtered_mask_infos)
            + len(quality_filtered_mask_infos)
            + len(fragment_filtered_mask_infos)
        ),
    )
    summary["dense_roi_refine_added"] = roi_refine_added
    summary["dense_roi_refine_regions"] = roi_refine_regions
    summary["dense_roi_refine_boxes"] = [list(box) for box in roi_boxes]
    summary["regular_dense_cluster_ids"] = regular_dense_cluster_ids
    summary["regular_dense_overcount_cluster_ids"] = regular_dense_overcount_cluster_ids
    summary["class_prior_filtered"] = len(prior_filtered_mask_infos)
    summary["physical_item_fragment_filtered"] = len(fragment_filtered_mask_infos)
    summary["physical_item_groups"] = physical_item_groups

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
