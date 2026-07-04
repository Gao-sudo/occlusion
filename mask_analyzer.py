"""Mask geometry analysis for occlusion counting.

Provides clustering of instance masks along hooks/stacks and 1-D projection
of masks along their principal axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from occlusion.config import (
    CLUSTER_EPS_PX,
    CLUSTER_MIN_SAMPLES,
    UNCOUNTABLE_DENSITY_MASKS_PER_M,
    UNCOUNTABLE_MASK_IOU_THRESHOLD,
    UNCOUNTABLE_MIN_CONFIDENCE_RATIO,
    UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE,
    UNCOUNTABLE_STEP_RATIO_THRESHOLD,
)


@dataclass
class MaskInfo:
    """Info for a single instance mask."""
    mask: np.ndarray          # bool HxW
    class_id: int
    class_name: str
    confidence: float
    source_index: int
    # centroid in pixel coords
    cx: float
    cy: float
    # bounding box
    x1: int
    y1: int
    x2: int
    y2: int
    # mask area in pixels
    area_px: int
    # principal orientation angle in degrees
    orientation_deg: float
    # context-aware decision
    decision: str = "unknown"
    decision_reasons: list[str] = field(default_factory=list)


@dataclass
class ClusterInfo:
    """A group of masks belonging to the same hook / stack."""
    cluster_id: int
    masks: list[MaskInfo]
    # axis direction (dx, dy) normalized, pointing along the stack/hook
    axis_direction: tuple[float, float]
    # approximate center of the cluster
    center: tuple[float, float]
    # countable: instance masks are well separated, use visible count directly
    # uncountable: severe overlap or dense packing, estimate via depth density
    countability: Literal["countable", "uncountable"] = "countable"
    countability_reasons: list[str] = field(default_factory=list)
    # dominant SKU class in this cluster (used for occlusion inheritance)
    dominant_class_id: int | None = None
    dominant_class_name: str | None = None
    # source indices of masks with high confidence (helper for decision engine)
    high_conf_source_indices: set[int] = field(default_factory=set)


def extract_mask_infos(
    masks: np.ndarray,           # NxHxW bool or uint8
    class_ids: np.ndarray,       # N
    class_names: list[str],
    confidences: np.ndarray | None = None,
) -> list[MaskInfo]:
    """Convert raw segmentation outputs to structured MaskInfo list."""
    if confidences is None:
        confidences = np.ones(len(masks), dtype=float)

    infos: list[MaskInfo] = []
    for i in range(len(masks)):
        mask = masks[i].astype(bool)
        if not mask.any():
            continue
        class_id = int(class_ids[i]) if i < len(class_ids) else 0
        # Skip if class_id is out of range (e.g. COCO pretrained on custom dataset)
        if class_id < 0 or class_id >= len(class_names):
            continue
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cx, cy = float(xs.mean()), float(ys.mean())
        area = int(mask.sum())

        # Principal orientation via moments
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["mu20"] + moments["mu02"] > 1e-6:
            orientation = 0.5 * np.arctan2(
                2 * moments["mu11"],
                moments["mu20"] - moments["mu02"],
            )
            orientation_deg = np.degrees(orientation)
        else:
            orientation_deg = 0.0

        infos.append(
            MaskInfo(
                mask=mask,
                class_id=class_id,
                class_name=class_names[class_id],
                confidence=float(confidences[i]) if confidences is not None and i < len(confidences) else 1.0,
                source_index=i,
                cx=cx,
                cy=cy,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                area_px=area,
                orientation_deg=orientation_deg,
            )
        )
    return infos


def is_top_horizontal_display_mask(
    mask_info: MaskInfo,
    image_shape: tuple[int, int],
    min_area_ratio: float = 0.035,
    max_center_y_ratio: float = 0.28,
    max_abs_orientation_deg: float = 30.0,
) -> bool:
    """Heuristic filter for large horizontal display signs above hanging products."""
    h, w = image_shape[:2]
    image_area = max(1, h * w)
    area_ratio = mask_info.area_px / image_area
    center_y_ratio = mask_info.cy / max(1, h)

    return (
        area_ratio >= min_area_ratio
        and center_y_ratio <= max_center_y_ratio
        and abs(mask_info.orientation_deg) <= max_abs_orientation_deg
    )


def filter_top_horizontal_display_masks(
    masks: list[MaskInfo],
    image_shape: tuple[int, int],
) -> tuple[list[MaskInfo], list[MaskInfo]]:
    """Remove display-board false positives before clustering/counting."""
    kept: list[MaskInfo] = []
    filtered: list[MaskInfo] = []
    for mask_info in masks:
        if is_top_horizontal_display_mask(mask_info, image_shape):
            filtered.append(mask_info)
        else:
            kept.append(mask_info)
    return kept, filtered


def cluster_masks(
    masks: list[MaskInfo],
    eps_px: float = CLUSTER_EPS_PX,
    min_samples: int = CLUSTER_MIN_SAMPLES,
) -> list[ClusterInfo]:
    """Group masks that belong to the same physical hook/stack using DBSCAN.

    Clustering is done on mask centroids.  The resulting axis direction is
    estimated as the principal eigenvector of the centroid distribution.
    """
    if not masks:
        return []

    centroids = np.array([[m.cx, m.cy] for m in masks])
    clustering = DBSCAN(eps=eps_px, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_

    clusters: list[ClusterInfo] = []
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        if lbl == -1:
            # noise: each isolated mask becomes its own cluster
            noise_indices = np.where(labels == -1)[0]
            for idx in noise_indices:
                m = masks[idx]
                clusters.append(
                    ClusterInfo(
                        cluster_id=len(clusters),
                        masks=[m],
                        axis_direction=(0.0, 1.0),  # default vertical
                        center=(m.cx, m.cy),
                    )
                )
            continue

        indices = np.where(labels == lbl)[0]
        cluster_masks_list = [masks[i] for i in indices]
        pts = centroids[indices]
        center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

        # Principal axis of centroid distribution
        axis = (0.0, 1.0)
        if len(pts) >= 2:
            cov = np.cov(pts.T)
            if cov.ndim == 2 and cov.shape == (2, 2) and not np.isnan(cov).any():
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    principal = eigvecs[:, np.argmax(eigvals)]
                    axis = (float(principal[0]), float(principal[1]))
                    norm = np.hypot(axis[0], axis[1]) + 1e-9
                    axis = (axis[0] / norm, axis[1] / norm)
                except Exception:
                    axis = (0.0, 1.0)

        clusters.append(
            ClusterInfo(
                cluster_id=len(clusters),
                masks=cluster_masks_list,
                axis_direction=axis,
                center=center,
            )
        )
    return clusters


def resolve_cluster_dominant_sku(cluster: ClusterInfo) -> ClusterInfo:
    """Pick the SKU class of the highest-confidence mask as the cluster dominant SKU.

    This is used to assign SKU to occluded/low-confidence instances in the same
    hook/stack by context inheritance.
    """
    if not cluster.masks:
        cluster.dominant_class_id = None
        cluster.dominant_class_name = None
        return cluster
    best = max(cluster.masks, key=lambda m: m.confidence)
    cluster.dominant_class_id = best.class_id
    cluster.dominant_class_name = best.class_name
    cluster.high_conf_source_indices = {
        m.source_index for m in cluster.masks
        if m.confidence >= 0.80
    }
    return cluster


def is_mask_axis_aligned(
    mask_info: MaskInfo,
    cluster: ClusterInfo,
    tolerance_deg: float = 30.0,
) -> bool:
    """Check whether a mask centroid lies close to the cluster's principal axis."""
    ax, ay = cluster.axis_direction
    if abs(ax) < 1e-6 and abs(ay) < 1e-6:
        return False

    cx, cy = cluster.center
    dx = mask_info.cx - cx
    dy = mask_info.cy - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return True

    # Angle between vector from cluster center to mask centroid and cluster axis
    dot = dx * ax + dy * ay
    norm = np.hypot(dx, dy)
    if norm <= 0:
        return True
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return angle_deg <= tolerance_deg


def build_cluster_by_source_index(clusters: list[ClusterInfo]) -> dict[int, ClusterInfo]:
    """Map each mask source_index to its containing cluster."""
    mapping: dict[int, ClusterInfo] = {}
    for cluster in clusters:
        for mask_info in cluster.masks:
            mapping[mask_info.source_index] = cluster
    return mapping


def project_depth_along_axis(
    depth_map: np.ndarray,
    cluster: ClusterInfo,
    projection_width_px: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a 1-D depth profile along the cluster's principal axis.

    Returns:
        positions: 1-D array of pixel positions along the axis
        depths:    corresponding median depth values
    """
    if not cluster.masks:
        return np.array([]), np.array([])

    # Build a combined ROI from all masks in the cluster
    combined_mask = np.zeros(depth_map.shape, dtype=bool)
    for m in cluster.masks:
        combined_mask |= m.mask

    ys, xs = np.where(combined_mask)
    if len(xs) == 0:
        return np.array([]), np.array([])

    # Project each point onto the axis through the cluster center
    ax, ay = cluster.axis_direction
    cx, cy = cluster.center

    # Coordinates relative to center
    dx = xs - cx
    dy = ys - cy
    # Scalar projection (signed distance along axis)
    proj = dx * ax + dy * ay

    # Sort along axis
    order = np.argsort(proj)
    proj_sorted = proj[order]
    depths_sorted = depth_map[ys[order], xs[order]]

    # Smooth / bin to reduce noise
    bin_size = max(1, int(np.ceil(len(proj_sorted) / 200)))  # target ~200 samples
    if bin_size <= 1:
        return proj_sorted, depths_sorted

    positions = []
    depths = []
    for i in range(0, len(proj_sorted), bin_size):
        positions.append(float(proj_sorted[i : i + bin_size].mean()))
        depths.append(float(np.median(depths_sorted[i : i + bin_size])))

    return np.array(positions), np.array(depths)


def detect_depth_steps(
    positions: np.ndarray,
    depths: np.ndarray,
    step_threshold_m: float,
    min_step_length_px: float = 5.0,
) -> list[tuple[float, float]]:
    """Detect depth discontinuities (steps) along the 1-D profile.

    Each returned tuple is (position_start, position_end) of a plateau.
    The number of plateaus corresponds to the number of visible+inferred items.
    """
    if len(depths) < 3:
        return []

    # Compute gradient
    grad = np.abs(np.diff(depths))
    # Mark step boundaries where gradient exceeds threshold
    is_step = grad > step_threshold_m

    # Find contiguous plateau segments
    plateaus: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(is_step)):
        if is_step[i]:
            if i - start >= 1:
                plateaus.append((start, i))
            start = i + 1
    if start < len(depths) - 1:
        plateaus.append((start, len(depths) - 1))

    # Filter by minimum length
    result = []
    for s, e in plateaus:
        if positions[e] - positions[s] >= min_step_length_px:
            result.append((float(positions[s]), float(positions[e])))

    return result



def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = float(np.logical_and(m1, m2).sum())
    union = float(np.logical_or(m1, m2).sum())
    if union <= 0:
        return 0.0
    return intersection / union


def max_intra_cluster_mask_iou(masks: list[MaskInfo]) -> float:
    """Return the maximum pairwise IoU among masks in a cluster."""
    n = len(masks)
    if n < 2:
        return 0.0
    max_iou = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            max_iou = max(max_iou, mask_iou(masks[i].mask, masks[j].mask))
    return max_iou


def classify_cluster_countability(
    cluster: ClusterInfo,
    depth_map: np.ndarray,
    mask_iou_threshold: float = UNCOUNTABLE_MASK_IOU_THRESHOLD,
    step_ratio_threshold: float = UNCOUNTABLE_STEP_RATIO_THRESHOLD,
    density_masks_per_m: float = UNCOUNTABLE_DENSITY_MASKS_PER_M,
    min_confidence_ratio: float = UNCOUNTABLE_MIN_CONFIDENCE_RATIO,
    min_visible_for_reference: int = UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE,
) -> ClusterInfo:
    """Classify a cluster as countable or uncountable based on geometry and depth.

    Rules (any match -> uncountable):
        1. Masks inside the cluster overlap too much (severe occlusion).
        2. Number of depth steps greatly exceeds visible count (stacked behind each other).
        3. Masks are extremely dense along the depth axis.
        4. Too many low-confidence detections in the cluster.
    """
    visible_count = len(cluster.masks)
    reasons: list[str] = []

    if visible_count < min_visible_for_reference:
        # Single isolated masks are countable by default
        cluster.countability = "countable"
        cluster.countability_reasons = []
        return cluster

    # Rule 1: mask overlap
    max_iou = max_intra_cluster_mask_iou(cluster.masks)
    if max_iou > mask_iou_threshold:
        reasons.append(f"high_mask_overlap_iou_{max_iou:.2f}")

    # Rule 2 & 3 need depth projection
    positions, depths = project_depth_along_axis(depth_map, cluster)
    if len(depths) > 0:
        depth_range = float(depths.max() - depths.min())
        steps = detect_depth_steps(positions, depths, step_threshold_m=0.015)
        step_count = len(steps)

        if visible_count > 0 and step_count / visible_count > step_ratio_threshold:
            reasons.append(f"steps_exceed_visible_{step_count}/{visible_count}")

        if depth_range > 0 and visible_count / depth_range > density_masks_per_m:
            reasons.append(f"high_density_{visible_count / depth_range:.1f}_masks_per_m")

    # Rule 4: low confidence ratio
    low_conf_count = sum(1 for m in cluster.masks if m.confidence < 0.5)
    if visible_count > 0 and low_conf_count / visible_count > (1.0 - min_confidence_ratio):
        reasons.append(f"low_confidence_ratio_{low_conf_count}/{visible_count}")

    if reasons:
        cluster.countability = "uncountable"
        cluster.countability_reasons = reasons
    else:
        cluster.countability = "countable"
        cluster.countability_reasons = []

    return cluster


def classify_clusters_countability(
    clusters: list[ClusterInfo],
    depth_map: np.ndarray,
    mask_iou_threshold: float = UNCOUNTABLE_MASK_IOU_THRESHOLD,
    step_ratio_threshold: float = UNCOUNTABLE_STEP_RATIO_THRESHOLD,
    density_masks_per_m: float = UNCOUNTABLE_DENSITY_MASKS_PER_M,
    min_confidence_ratio: float = UNCOUNTABLE_MIN_CONFIDENCE_RATIO,
    min_visible_for_reference: int = UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE,
) -> list[ClusterInfo]:
    """Classify countability for all clusters."""
    return [
        classify_cluster_countability(
            c,
            depth_map,
            mask_iou_threshold=mask_iou_threshold,
            step_ratio_threshold=step_ratio_threshold,
            density_masks_per_m=density_masks_per_m,
            min_confidence_ratio=min_confidence_ratio,
            min_visible_for_reference=min_visible_for_reference,
        )
        for c in clusters
    ]
