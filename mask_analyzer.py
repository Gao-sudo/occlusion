"""Mask geometry analysis for occlusion counting.

Provides clustering of instance masks along hooks/stacks and 1-D projection
of masks along their principal axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import cv2
import numpy as np

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
    dominant_class_id: Optional[int] = None
    dominant_class_name: Optional[str] = None
    # source indices of masks with high confidence (helper for decision engine)
    high_conf_source_indices: set[int] = field(default_factory=set)


def extract_mask_infos(
    masks: np.ndarray,           # NxHxW bool or uint8
    class_ids: np.ndarray,       # N
    class_names: list[str],
    confidences: Optional[np.ndarray] = None,
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
    max_center_y_ratio: float = 0.20,
    max_abs_orientation_deg: float = 30.0,
    max_confidence: float = 0.60,
    min_bbox_aspect: float = 2.5,
) -> bool:
    """Heuristic filter for large horizontal display signs above hanging products."""
    h, w = image_shape[:2]
    image_area = max(1, h * w)
    area_ratio = mask_info.area_px / image_area
    center_y_ratio = mask_info.cy / max(1, h)
    bbox_aspect = (mask_info.x2 - mask_info.x1) / max(1, mask_info.y2 - mask_info.y1)

    return (
        area_ratio >= min_area_ratio
        and center_y_ratio <= max_center_y_ratio
        and abs(mask_info.orientation_deg) <= max_abs_orientation_deg
        and mask_info.confidence <= max_confidence
        and bbox_aspect >= min_bbox_aspect
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


def _bbox_area(mask_info: MaskInfo) -> float:
    return float(max(0, mask_info.x2 - mask_info.x1) * max(0, mask_info.y2 - mask_info.y1))


def _bbox_intersection_area(a: MaskInfo, b: MaskInfo) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_iou(a: MaskInfo, b: MaskInfo) -> float:
    inter = _bbox_intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_overlap_ratios(a: MaskInfo, b: MaskInfo) -> tuple[float, float]:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    x_overlap = max(0, x2 - x1)
    y_overlap = max(0, y2 - y1)
    min_w = max(1, min(a.x2 - a.x1, b.x2 - b.x1))
    min_h = max(1, min(a.y2 - a.y1, b.y2 - b.y1))
    return x_overlap / min_w, y_overlap / min_h


def _bbox_center_distance(a: MaskInfo, b: MaskInfo) -> float:
    return float(np.hypot(a.cx - b.cx, a.cy - b.cy))


def _bbox_centers_are_close(a: MaskInfo, b: MaskInfo, y_factor: float = 0.15, x_factor: float = 0.25) -> bool:
    min_w = max(1, min(a.x2 - a.x1, b.x2 - b.x1))
    min_h = max(1, min(a.y2 - a.y1, b.y2 - b.y1))
    return abs(a.cx - b.cx) <= x_factor * min_w and abs(a.cy - b.cy) <= y_factor * min_h


def load_class_priors(path: Path | Optional[str]) -> dict[str, object]:
    """Load optional class geometry priors generated from training labels."""
    if path is None:
        return {}
    prior_path = Path(path)
    if not prior_path.exists():
        return {}
    try:
        data = json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def filter_instances_by_class_priors(
    masks: list[MaskInfo],
    image_shape: tuple[int, int],
    class_priors: Optional[Dict[str, object]] = None,
    min_samples: int = 20,
) -> tuple[list[MaskInfo], list[MaskInfo]]:
    """Filter geometry outliers using class-level label statistics.

    The thresholds are intentionally loose; this catches obvious fragments while
    avoiding aggressive pruning of rare or partially occluded SKUs.
    """
    if not class_priors:
        return masks, []
    classes = class_priors.get("classes")
    if not isinstance(classes, dict):
        return masks, []

    h, w = image_shape[:2]
    image_area = max(1, h * w)
    kept: list[MaskInfo] = []
    filtered: list[MaskInfo] = []
    for mask_info in masks:
        prior = classes.get(str(mask_info.class_id))
        if not isinstance(prior, dict) or int(prior.get("count", 0)) < min_samples:
            kept.append(mask_info)
            continue

        area_prior = prior.get("bbox_area")
        aspect_prior = prior.get("aspect_h_over_w")
        if not isinstance(area_prior, dict) or not isinstance(aspect_prior, dict):
            kept.append(mask_info)
            continue

        bbox_area_norm = _bbox_area(mask_info) / image_area
        bbox_w = max(1, mask_info.x2 - mask_info.x1)
        bbox_h = max(1, mask_info.y2 - mask_info.y1)
        aspect = bbox_h / bbox_w
        p01_area = float(area_prior.get("p01", 0.0))
        p05_aspect = float(aspect_prior.get("p05", 0.0))
        p95_aspect = float(aspect_prior.get("p95", 0.0))

        too_tiny = p01_area > 0 and bbox_area_norm < p01_area * 0.35 and mask_info.confidence < 0.70
        aspect_low = p05_aspect > 0 and aspect < p05_aspect * 0.35 and mask_info.confidence < 0.55
        aspect_high = p95_aspect > 0 and aspect > p95_aspect * 2.8 and mask_info.confidence < 0.55
        if too_tiny or aspect_low or aspect_high:
            mask_info.decision = "filtered"
            reason = "class_prior_geometry_outlier"
            if too_tiny:
                reason = "class_prior_tiny_fragment"
            mask_info.decision_reasons = list(mask_info.decision_reasons) + [reason]
            filtered.append(mask_info)
        else:
            kept.append(mask_info)

    return kept, filtered


def _should_merge_physical_item(a: MaskInfo, b: MaskInfo) -> bool:
    if a.class_id != b.class_id:
        return False
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    if area_a <= 0 or area_b <= 0:
        return False

    inter = _bbox_intersection_area(a, b)
    containment = inter / max(1.0, min(area_a, area_b))
    iou = _bbox_iou(a, b)
    center_distance = _bbox_center_distance(a, b)
    max_w = max(a.x2 - a.x1, b.x2 - b.x1, 1)
    max_h = max(a.y2 - a.y1, b.y2 - b.y1, 1)
    max_diag = float(np.hypot(max_w, max_h))

    if iou >= 0.50:
        return True
    if containment >= 0.88 and center_distance <= 0.45 * max_diag:
        return True
    if containment >= 0.72 and min(area_a, area_b) <= 0.35 * max(area_a, area_b):
        return True

    return False


def merge_physical_item_fragments(
    masks: list[MaskInfo],
) -> tuple[list[MaskInfo], list[MaskInfo], list[list[int]]]:
    """Merge graph-connected mask fragments that likely describe one item."""
    if len(masks) < 2:
        return masks, [], []

    parent = list(range(len(masks)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _should_merge_physical_item(masks[i], masks[j]):
                union(i, j)

    groups_by_root: dict[int, list[int]] = {}
    for idx in range(len(masks)):
        groups_by_root.setdefault(find(idx), []).append(idx)

    kept: list[MaskInfo] = []
    filtered: list[MaskInfo] = []
    merged_groups: list[list[int]] = []
    for group in groups_by_root.values():
        if len(group) == 1:
            kept.append(masks[group[0]])
            continue
        group_masks = [masks[idx] for idx in group]
        representative = max(group_masks, key=lambda m: (m.confidence, m.area_px))
        merged_groups.append(sorted(m.source_index for m in group_masks))
        kept.append(representative)
        for mask_info in group_masks:
            if mask_info is representative:
                continue
            mask_info.decision = "filtered"
            mask_info.decision_reasons = list(mask_info.decision_reasons) + ["physical_item_fragment"]
            filtered.append(mask_info)

    kept.sort(key=lambda m: m.source_index)
    filtered.sort(key=lambda m: m.source_index)
    merged_groups.sort(key=lambda group: group[0])
    return kept, filtered, merged_groups


def filter_counting_instances(
    masks: list[MaskInfo],
    cluster_by_source_index: Optional[Dict[int, ClusterInfo]] = None,
    min_context_confidence: float = 0.40,
    same_class_containment_threshold: float = 0.85,
    axis_only_unknown_confidence: float = 0.70,
    axis_only_unknown_class_base: int = 1,
    axis_only_unknown_class_sqrt_factor: float = 1.0,
) -> tuple[list[MaskInfo], list[MaskInfo]]:
    """Filter low-quality instances before counting.

    This is intentionally conservative:
    - low-confidence context-only detections are moved to filtered;
    - same-class detections mostly contained in a higher-confidence detection
      are treated as duplicate fragments.
    """
    candidates: list[MaskInfo] = []
    filtered: list[MaskInfo] = []

    for mask_info in masks:
        cluster = cluster_by_source_index.get(mask_info.source_index) if cluster_by_source_index else None
        dense_cluster = cluster is not None and cluster.countability == "uncountable"

        weak_unknown_axis_only = (
            mask_info.decision == "unknown"
            and mask_info.decision_reasons == ["aligned_with_cluster_axis"]
            and mask_info.confidence < axis_only_unknown_confidence
        )
        if weak_unknown_axis_only and not dense_cluster:
            mask_info.decision = "filtered"
            mask_info.decision_reasons = list(mask_info.decision_reasons) + ["weak_unknown_axis_only"]
            filtered.append(mask_info)
        elif (
            mask_info.decision == "confirmed_by_context"
            and mask_info.confidence < min_context_confidence
            and not dense_cluster
        ):
            mask_info.decision = "filtered"
            mask_info.decision_reasons = list(mask_info.decision_reasons) + ["low_context_confidence"]
            filtered.append(mask_info)
        else:
            candidates.append(mask_info)

    kept: list[MaskInfo] = []
    for mask_info in sorted(candidates, key=lambda m: m.confidence, reverse=True):
        cluster = cluster_by_source_index.get(mask_info.source_index) if cluster_by_source_index else None
        dense_cluster = cluster is not None and cluster.countability == "uncountable"
        duplicate = False
        for kept_info in kept:
            if mask_info.class_id != kept_info.class_id:
                continue
            kept_cluster = cluster_by_source_index.get(kept_info.source_index) if cluster_by_source_index else None
            if kept_cluster is not None and cluster is not None and kept_cluster.cluster_id != cluster.cluster_id:
                continue
            if _bbox_area(mask_info) > _bbox_area(kept_info):
                continue
            containment = _bbox_intersection_area(mask_info, kept_info) / max(_bbox_area(mask_info), 1.0)
            duplicate_threshold = 0.97 if dense_cluster else same_class_containment_threshold
            if containment >= duplicate_threshold:
                mask_info.decision = "filtered"
                mask_info.decision_reasons = list(mask_info.decision_reasons) + ["same_class_duplicate_fragment"]
                filtered.append(mask_info)
                duplicate = True
                break
        if not duplicate:
            kept.append(mask_info)

    decision_priority = {
        "confirmed": 3,
        "confirmed_by_context": 2,
        "unknown": 1,
        "filtered": 0,
    }
    cross_class_duplicate_ids: set[int] = set()
    confirmed_scene_count = sum(1 for mask_info in kept if mask_info.decision == "confirmed")
    for i, mask_info in enumerate(kept):
        if mask_info.source_index in cross_class_duplicate_ids:
            continue
        for other_info in kept[i + 1 :]:
            if other_info.source_index in cross_class_duplicate_ids:
                continue
            if mask_info.class_id == other_info.class_id:
                continue
            cross_class_iou = _bbox_iou(mask_info, other_info)
            cross_class_containment = _bbox_intersection_area(mask_info, other_info) / max(
                1.0,
                min(_bbox_area(mask_info), _bbox_area(other_info)),
            )
            if cross_class_iou < 0.93 and cross_class_containment < 0.90:
                continue

            mask_rank = (
                decision_priority.get(mask_info.decision, 0),
                mask_info.confidence,
                _bbox_area(mask_info),
            )
            other_rank = (
                decision_priority.get(other_info.decision, 0),
                other_info.confidence,
                _bbox_area(other_info),
            )
            winner, loser = (mask_info, other_info) if mask_rank >= other_rank else (other_info, mask_info)
            low_count_weak_scene_duplicate = len(kept) <= 2 and confirmed_scene_count == 0
            context_duplicate = (
                winner.decision == "confirmed_by_context"
                and loser.decision in {"confirmed_by_context", "unknown"}
                and cross_class_containment >= 0.90
                and _bbox_centers_are_close(mask_info, other_info)
            )
            confirmed_duplicate = winner.decision == "confirmed" and cross_class_iou >= 0.93
            if not confirmed_duplicate and not low_count_weak_scene_duplicate and not context_duplicate:
                continue

            loser.decision = "filtered"
            loser.decision_reasons = list(loser.decision_reasons) + ["cross_class_duplicate_bbox"]
            filtered.append(loser)
            cross_class_duplicate_ids.add(loser.source_index)

    if cross_class_duplicate_ids:
        kept = [mask_info for mask_info in kept if mask_info.source_index not in cross_class_duplicate_ids]

    class_support: dict[int, int] = {}
    axis_only_unknown_by_class: dict[int, list[MaskInfo]] = {}
    for mask_info in kept:
        axis_only_unknown = (
            mask_info.decision == "unknown"
            and mask_info.decision_reasons == ["aligned_with_cluster_axis"]
        )
        if axis_only_unknown:
            axis_only_unknown_by_class.setdefault(mask_info.class_id, []).append(mask_info)
        else:
            class_support[mask_info.class_id] = class_support.get(mask_info.class_id, 0) + 1

    capped_axis_only_unknown: set[int] = set()
    for class_id, unknown_masks in axis_only_unknown_by_class.items():
        support = class_support.get(class_id, 0)
        cap = axis_only_unknown_class_base + int(math.floor(axis_only_unknown_class_sqrt_factor * math.sqrt(support)))
        if len(unknown_masks) <= cap:
            continue

        keep_ids = {
            mask_info.source_index
            for mask_info in sorted(unknown_masks, key=lambda m: m.confidence, reverse=True)[:cap]
        }
        for mask_info in unknown_masks:
            if mask_info.source_index in keep_ids:
                continue
            mask_info.decision = "filtered"
            mask_info.decision_reasons = list(mask_info.decision_reasons) + ["axis_only_unknown_class_cap"]
            filtered.append(mask_info)
            capped_axis_only_unknown.add(mask_info.source_index)

    if capped_axis_only_unknown:
        kept = [mask_info for mask_info in kept if mask_info.source_index not in capped_axis_only_unknown]

    kept.sort(key=lambda m: m.source_index)
    filtered.sort(key=lambda m: m.source_index)
    return kept, filtered


def _cluster_centroids_fallback(
    centroids: np.ndarray,
    eps_px: float,
    min_samples: int,
) -> np.ndarray:
    n = len(centroids)
    if n == 0:
        return np.array([], dtype=int)

    labels = np.full(n, -1, dtype=int)
    if n == 1:
        labels[0] = 0 if min_samples <= 1 else -1
        return labels

    deltas = centroids[:, None, :] - centroids[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    neighbors = [np.where(distances[i] <= eps_px)[0] for i in range(n)]

    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        if len(neighbors[start]) < min_samples:
            continue

        stack = [start]
        component: set[int] = set()
        while stack:
            idx = stack.pop()
            if idx in component:
                continue
            component.add(idx)
            for neighbor in neighbors[idx]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                if len(neighbors[neighbor]) >= min_samples and neighbor not in component:
                    stack.append(int(neighbor))
                else:
                    component.add(int(neighbor))

        for idx in component:
            labels[idx] = cluster_id
        cluster_id += 1

    return labels


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
    labels = _cluster_centroids_fallback(centroids, eps_px=eps_px, min_samples=min_samples)

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
            try:
                deltas = pts - pts.mean(axis=0)
                lengths = np.linalg.norm(deltas, axis=1)
                principal = deltas[np.argmax(lengths)]
                if np.allclose(principal, 0.0):
                    principal = pts[-1] - pts[0]
                norm = np.hypot(float(principal[0]), float(principal[1]))
                if norm > 1e-9:
                    axis = (float(principal[0] / norm), float(principal[1] / norm))
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


def _count_axis_slots(cluster: ClusterInfo) -> int:
    if not cluster.masks:
        return 0
    ax, ay = cluster.axis_direction
    if abs(ax) < 1e-9 and abs(ay) < 1e-9:
        return len(cluster.masks)

    center_x, center_y = cluster.center
    spans: list[tuple[float, float]] = []
    for mask_info in cluster.masks:
        dx = mask_info.cx - center_x
        dy = mask_info.cy - center_y
        projection = dx * ax + dy * ay
        extent = max(mask_info.x2 - mask_info.x1, mask_info.y2 - mask_info.y1)
        half_span = max(8.0, float(extent) * 0.35)
        spans.append((projection - half_span, projection + half_span))

    spans.sort(key=lambda item: item[0])
    slots = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            slots += 1
            current_start, current_end = start, end
    slots += 1
    return max(1, slots)


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
    single_class_mode = cluster.dominant_class_name == "countable_product"

    axis_slot_count = _count_axis_slots(cluster)

    if visible_count < min_visible_for_reference and not (
        single_class_mode and visible_count >= 2 and axis_slot_count < visible_count
    ):
        # Single isolated masks are countable by default
        cluster.countability = "countable"
        cluster.countability_reasons = []
        return cluster

    # Rule 1: mask overlap
    max_iou = max_intra_cluster_mask_iou(cluster.masks)
    if max_iou > mask_iou_threshold:
        reasons.append(f"high_mask_overlap_iou_{max_iou:.2f}")

    if single_class_mode and visible_count >= 3 and axis_slot_count <= max(1, visible_count - 1):
        reasons.append(f"single_class_dense_slots_{axis_slot_count}/{visible_count}")

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
    low_conf_ratio = low_conf_count / visible_count if visible_count > 0 else 0.0
    if visible_count > 0 and low_conf_ratio > (1.0 - min_confidence_ratio):
        reasons.append(f"low_confidence_ratio_{low_conf_count}/{visible_count}")

    if single_class_mode and visible_count >= 4 and low_conf_ratio >= 0.25:
        reasons.append(f"single_class_low_conf_dense_{low_conf_count}/{visible_count}")

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
