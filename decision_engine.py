"""Context-aware decision engine for occlusion counting.

Assigns each instance a decision label:
    - confirmed: high-confidence, clear detection.
    - confirmed_by_context: lower confidence, but supported by cluster/geometry context.
    - unknown: lower confidence and no supporting context.
    - filtered: removed by upstream filters (e.g. horizontal signboard).

This module intentionally does not use depth, because the monocular depth model
is unreliable for side-by-side hanging products in the current setup.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from occlusion.config import (
    DECISION_CLUSTER_AXIS_ALIGN_DEG,
    DECISION_CONFIRMED_THRESHOLD,
    DECISION_CONTEXT_MAX_CONF,
    DECISION_CONTEXT_MIN_CONF,
    DECISION_VERTICAL_ASPECT_MIN,
)
from occlusion.mask_analyzer import ClusterInfo, MaskInfo, is_mask_axis_aligned


def _is_inside_roi(mask_info: MaskInfo, roi: tuple[int, int, int, int] | Sequence[int]) -> bool:
    """Check whether mask centroid is inside a bounding-box ROI (x1, y1, x2, y2)."""
    if roi is None or len(roi) != 4:
        return False
    x1, y1, x2, y2 = roi
    return x1 <= mask_info.cx <= x2 and y1 <= mask_info.cy <= y2


def _has_vertical_shape(mask_info: MaskInfo, min_aspect: float = DECISION_VERTICAL_ASPECT_MIN) -> bool:
    """Check whether mask is vertically elongated and roughly upright."""
    width = max(1, mask_info.x2 - mask_info.x1)
    height = max(1, mask_info.y2 - mask_info.y1)
    aspect = height / width
    if aspect < min_aspect:
        return False
    # Orientation is defined such that 0 deg = horizontal, 90 deg = vertical.
    # Allow ±30 deg around vertical.
    return abs(abs(mask_info.orientation_deg) - 90.0) <= 30.0


def _same_cluster_has_high_conf_instance(
    mask_info: MaskInfo,
    cluster: ClusterInfo | None,
) -> bool:
    if cluster is None:
        return False
    return mask_info.source_index in cluster.high_conf_source_indices or len(cluster.high_conf_source_indices) > 0


def _is_part_of_product_cluster(mask_info: MaskInfo, cluster: ClusterInfo | None) -> bool:
    if cluster is None:
        return False
    # Multi-mask cluster with roughly vertical axis is treated as a product hook/stack.
    if len(cluster.masks) < 2:
        return False
    ax, ay = cluster.axis_direction
    # Vertical axis: y component dominates
    return abs(ay) >= abs(ax)


def classify_instance_decision(
    mask_info: MaskInfo,
    cluster: ClusterInfo | None,
    image_shape: tuple[int, int],
    display_roi: tuple[int, int, int, int] | Sequence[int] | None = None,
    confirmed_threshold: float = DECISION_CONFIRMED_THRESHOLD,
    context_min_conf: float = DECISION_CONTEXT_MIN_CONF,
    context_max_conf: float = DECISION_CONTEXT_MAX_CONF,
) -> tuple[str, list[str]]:
    """Classify a single instance into a decision tier with supporting reasons."""
    conf = mask_info.confidence

    # Tier 1: high confidence detections
    if conf >= confirmed_threshold:
        return "confirmed", ["high_confidence"]

    reasons: list[str] = []

    # Evaluate context rules
    if _same_cluster_has_high_conf_instance(mask_info, cluster):
        reasons.append("same_cluster_as_high_conf_instance")
    if display_roi is not None and _is_inside_roi(mask_info, display_roi):
        reasons.append("inside_display_roi")
    if _has_vertical_shape(mask_info):
        reasons.append("vertical_product_shape")
    if cluster is not None and is_mask_axis_aligned(
        mask_info, cluster, tolerance_deg=DECISION_CLUSTER_AXIS_ALIGN_DEG
    ):
        reasons.append("aligned_with_cluster_axis")
    if _is_part_of_product_cluster(mask_info, cluster):
        reasons.append("part_of_product_cluster")

    # Tier 2: context-supported detections
    # Normal confidence range
    if context_min_conf <= conf < context_max_conf and reasons:
        return "confirmed_by_context", reasons

    # Very low confidence but strongly supported by cluster context
    if conf < context_min_conf and reasons:
        strong_context = (
            "same_cluster_as_high_conf_instance" in reasons
            and "vertical_product_shape" in reasons
            and "aligned_with_cluster_axis" in reasons
        )
        if strong_context:
            return "confirmed_by_context", reasons

    # Tier 3: no context support
    if reasons:
        return "unknown", reasons
    return "unknown", ["low_confidence_no_context"]


def classify_all_instance_decisions(
    mask_infos: list[MaskInfo],
    cluster_by_source_index: dict[int, ClusterInfo],
    image_shape: tuple[int, int],
    display_roi: tuple[int, int, int, int] | Sequence[int] | None = None,
) -> None:
    """In-place update of decision fields on all MaskInfo objects."""
    for mask_info in mask_infos:
        cluster = cluster_by_source_index.get(mask_info.source_index)
        decision, reasons = classify_instance_decision(
            mask_info, cluster, image_shape, display_roi=display_roi
        )
        mask_info.decision = decision
        mask_info.decision_reasons = reasons


def compute_decision_counts(masks: list[MaskInfo]) -> dict[str, int]:
    """Return counts per decision label."""
    counts = {"confirmed": 0, "confirmed_by_context": 0, "unknown": 0, "filtered": 0}
    for m in masks:
        counts[m.decision] = counts.get(m.decision, 0) + 1
    return counts
