"""Multi-modal fusion counting engine.

Combines instance-segmentation masks with monocular depth estimates to infer
the total number of items in a stack, including occluded ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from occlusion.config import (
    COUNT_DIFF_TOLERANCE,
    DEFAULT_SKU_SPEC,
    DEPTH_STEP_THRESHOLD,
)
from occlusion.mask_analyzer import ClusterInfo, detect_depth_steps, project_depth_along_axis


@dataclass
class CountResult:
    """Counting result for a single cluster (hook/stack)."""
    cluster_id: int
    class_name: str
    visible_count: int
    estimated_total: int
    occlusion_inferred: int
    confidence: Literal["high", "medium", "low"]
    depth_range_m: float | None = None
    unit_depth_m: float | None = None
    method: str = "unknown"
    # diagnostics
    depth_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_values: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_steps: list[tuple[float, float]] = field(default_factory=list)


def count_cluster(
    cluster: ClusterInfo,
    depth_map: np.ndarray,
    sku_specs: dict[str, dict[str, Any]] | None = None,
    step_threshold: float = DEPTH_STEP_THRESHOLD,
    count_tolerance: int = COUNT_DIFF_TOLERANCE,
) -> CountResult:
    """Estimate total item count for a single cluster using seg + depth fusion.

    Algorithm (from report Section 4.1):
        1. Count visible items from segmentation masks.
        2. Project depth along the cluster's principal axis.
        3. Detect depth steps (discontinuities) -> each step is a boundary between items.
        4. If depth is reliable, use total depth range / unit_depth as estimate.
        5. Fuse visible count and depth-based estimate; flag occlusion when they differ.
    """
    if sku_specs is None:
        sku_specs = DEFAULT_SKU_SPEC

    class_name = cluster.masks[0].class_name if cluster.masks else "unknown"
    visible_count = len(cluster.masks)

    # Default fallback
    result = CountResult(
        cluster_id=cluster.cluster_id,
        class_name=class_name,
        visible_count=visible_count,
        estimated_total=visible_count,
        occlusion_inferred=0,
        confidence="low",
    )

    if visible_count == 0:
        return result

    # --- Depth projection ---
    positions, depths = project_depth_along_axis(depth_map, cluster)
    if len(positions) == 0:
        return result

    result.depth_positions = positions
    result.depth_values = depths

    # --- Detect depth steps ---
    steps = detect_depth_steps(positions, depths, step_threshold_m=step_threshold)
    result.depth_steps = steps

    # --- Depth-range based estimate ---
    depth_min = float(depths.min())
    depth_max = float(depths.max())
    depth_range = depth_max - depth_min
    result.depth_range_m = depth_range

    spec = sku_specs.get(class_name, {})
    unit_depth = spec.get("unit_depth_m")
    result.unit_depth_m = unit_depth

    depth_based_count: int | None = None
    if unit_depth and unit_depth > 0:
        depth_based_count = max(1, int(round(depth_range / unit_depth)))

    # Step-based count (number of plateaus)
    step_based_count = max(1, len(steps)) if steps else None

    # --- Fusion decision ---
    if depth_based_count is not None and step_based_count is not None:
        # Both depth range and steps are informative
        # Prefer step-based if visible_count is close to it, else use conservative max
        if abs(visible_count - step_based_count) <= count_tolerance:
            result.estimated_total = visible_count
            result.confidence = "high"
            result.method = "seg+steps_agree"
        elif step_based_count > visible_count:
            result.estimated_total = step_based_count
            result.occlusion_inferred = step_based_count - visible_count
            result.confidence = "medium"
            result.method = "steps_infer_occlusion"
        elif depth_based_count > visible_count:
            result.estimated_total = depth_based_count
            result.occlusion_inferred = depth_based_count - visible_count
            result.confidence = "medium"
            result.method = "depth_range_infer_occlusion"
        else:
            result.estimated_total = visible_count
            result.confidence = "high"
            result.method = "visible_only"
    elif depth_based_count is not None:
        if depth_based_count > visible_count + count_tolerance:
            result.estimated_total = depth_based_count
            result.occlusion_inferred = depth_based_count - visible_count
            result.confidence = "medium"
            result.method = "depth_range_only"
        else:
            result.estimated_total = visible_count
            result.confidence = "high"
            result.method = "visible_only"
    elif step_based_count is not None:
        if step_based_count > visible_count + count_tolerance:
            result.estimated_total = step_based_count
            result.occlusion_inferred = step_based_count - visible_count
            result.confidence = "medium"
            result.method = "steps_only"
        else:
            result.estimated_total = visible_count
            result.confidence = "high"
            result.method = "visible_only"
    else:
        # No depth info -> fallback to visible count
        result.estimated_total = visible_count
        result.confidence = "low"
        result.method = "visible_fallback"

    return result


def count_all_clusters(
    clusters: list[ClusterInfo],
    depth_map: np.ndarray,
    sku_specs: dict[str, dict[str, Any]] | None = None,
    step_threshold: float = DEPTH_STEP_THRESHOLD,
    count_tolerance: int = COUNT_DIFF_TOLERANCE,
) -> list[CountResult]:
    """Run fusion counting on all clusters."""
    return [
        count_cluster(
            c,
            depth_map,
            sku_specs=sku_specs,
            step_threshold=step_threshold,
            count_tolerance=count_tolerance,
        )
        for c in clusters
    ]


def summarize_counts(results: list[CountResult]) -> dict[str, Any]:
    """Aggregate counts across all clusters for JSON output."""
    total_visible = sum(r.visible_count for r in results)
    total_estimated = sum(r.estimated_total for r in results)
    total_inferred = sum(r.occlusion_inferred for r in results)

    cluster_list = []
    for r in results:
        cluster_list.append(
            {
                "cluster_id": r.cluster_id,
                "class_name": r.class_name,
                "visible_count": r.visible_count,
                "estimated_total": r.estimated_total,
                "occlusion_inferred": r.occlusion_inferred,
                "confidence": r.confidence,
                "depth_range_m": r.depth_range_m,
                "unit_depth_m": r.unit_depth_m,
                "method": r.method,
            }
        )

    return {
        "total_visible": total_visible,
        "total_estimated": total_estimated,
        "total_occlusion_inferred": total_inferred,
        "clusters": cluster_list,
    }
