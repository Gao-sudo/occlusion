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
    UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE,
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
    confidence: str
    depth_range_m: float | None = None
    unit_depth_m: float | None = None
    learned_unit_depth_m: float | None = None
    method: str = "unknown"
    countability: str = "countable"
    countability_reasons: list[str] = field(default_factory=list)
    # context-aware decision counts within this cluster
    confirmed_count: int = 0
    confirmed_by_context_count: int = 0
    unknown_count: int = 0
    # dominant SKU inherited by cluster context
    dominant_class_id: int | None = None
    dominant_class_name: str | None = None
    # diagnostics
    depth_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_values: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_steps: list[tuple[float, float]] = field(default_factory=list)


def estimate_unit_depth_per_class(
    clusters: list[ClusterInfo],
    depth_map: np.ndarray,
    min_visible: int = UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE,
) -> dict[str, float]:
    """Learn unit depth per item for each SKU class from countable clusters.

    Since monocular depth has no absolute scale, we learn the relative
    depth footprint of one item from well-separated (countable) instances
    on the same image, then reuse it for dense/uncountable regions.
    """
    records: dict[str, list[float]] = {}

    for cluster in clusters:
        if cluster.countability != "countable":
            continue
        visible_count = len(cluster.masks)
        if visible_count < min_visible:
            continue

        positions, depths = project_depth_along_axis(depth_map, cluster)
        if len(depths) == 0:
            continue

        depth_range = float(depths.max() - depths.min())
        if depth_range <= 0:
            continue

        unit_depth = depth_range / visible_count
        class_name = cluster.masks[0].class_name if cluster.masks else "unknown"
        records.setdefault(class_name, []).append(unit_depth)

    return {cls: float(np.median(values)) for cls, values in records.items()}


def count_cluster(
    cluster: ClusterInfo,
    depth_map: np.ndarray,
    sku_specs: dict[str, dict[str, Any]] | None = None,
    unit_depth_map: dict[str, float] | None = None,
    step_threshold: float = DEPTH_STEP_THRESHOLD,
    count_tolerance: int = COUNT_DIFF_TOLERANCE,
) -> CountResult:
    """Estimate total item count for a single cluster using seg + depth fusion.

    Two tracks:
        - countable clusters: use classic visible count + depth step fusion.
        - uncountable clusters: use learned unit depth per item from countable
          clusters of the same class to estimate density.
    """
    if sku_specs is None:
        sku_specs = DEFAULT_SKU_SPEC

    class_name = cluster.masks[0].class_name if cluster.masks else "unknown"
    visible_count = len(cluster.masks)

    # Decision counts from context-aware classification
    decision_counts = {"confirmed": 0, "confirmed_by_context": 0, "unknown": 0, "filtered": 0}
    for m in cluster.masks:
        decision_counts[m.decision] = decision_counts.get(m.decision, 0) + 1

    # Default fallback
    result = CountResult(
        cluster_id=cluster.cluster_id,
        class_name=class_name,
        visible_count=visible_count,
        estimated_total=visible_count,
        occlusion_inferred=0,
        confidence="low",
        countability=cluster.countability,
        countability_reasons=list(cluster.countability_reasons),
        confirmed_count=decision_counts.get("confirmed", 0),
        confirmed_by_context_count=decision_counts.get("confirmed_by_context", 0),
        unknown_count=decision_counts.get("unknown", 0),
        dominant_class_id=cluster.dominant_class_id,
        dominant_class_name=cluster.dominant_class_name,
    )

    if visible_count == 0:
        return result

    # --- Depth projection ---
    positions, depths = project_depth_along_axis(depth_map, cluster)
    if len(positions) == 0:
        return result

    result.depth_positions = positions
    result.depth_values = depths

    depth_min = float(depths.min())
    depth_max = float(depths.max())
    depth_range = depth_max - depth_min
    result.depth_range_m = depth_range

    # --- Uncountable track: density estimation via learned unit depth ---
    if cluster.countability == "uncountable":
        learned_unit = unit_depth_map.get(class_name) if unit_depth_map else None
        result.learned_unit_depth_m = learned_unit

        if learned_unit and learned_unit > 0 and depth_range > 0:
            estimated = max(visible_count, int(round(depth_range / learned_unit)))
            result.estimated_total = estimated
            result.occlusion_inferred = estimated - visible_count
            result.confidence = "medium"
            result.method = "depth_density_estimate"
            result.unit_depth_m = learned_unit
        else:
            # No reference countable cluster for this class
            result.estimated_total = visible_count
            result.occlusion_inferred = 0
            result.confidence = "low"
            result.method = "uncountable_no_reference"
        return result

    # --- Countable track: classic fusion ---
    steps = detect_depth_steps(positions, depths, step_threshold_m=step_threshold)
    result.depth_steps = steps

    spec = sku_specs.get(class_name, {})
    unit_depth = spec.get("unit_depth_m")
    result.unit_depth_m = unit_depth

    depth_based_count: int | None = None
    if unit_depth and unit_depth > 0:
        depth_based_count = max(1, int(round(depth_range / unit_depth)))

    step_based_count = max(1, len(steps)) if steps else None

    # Fusion decision
    if depth_based_count is not None and step_based_count is not None:
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
        result.estimated_total = visible_count
        result.confidence = "low"
        result.method = "visible_fallback"

    return result


def count_all_clusters(
    clusters: list[ClusterInfo],
    depth_map: np.ndarray,
    sku_specs: dict[str, dict[str, Any]] | None = None,
    unit_depth_map: dict[str, float] | None = None,
    step_threshold: float = DEPTH_STEP_THRESHOLD,
    count_tolerance: int = COUNT_DIFF_TOLERANCE,
) -> list[CountResult]:
    """Run fusion counting on all clusters."""
    return [
        count_cluster(
            c,
            depth_map,
            sku_specs=sku_specs,
            unit_depth_map=unit_depth_map,
            step_threshold=step_threshold,
            count_tolerance=count_tolerance,
        )
        for c in clusters
    ]


def summarize_counts(
    results: list[CountResult],
    filtered_count: int = 0,
) -> dict[str, Any]:
    """Aggregate counts across all clusters for JSON output."""
    total_visible = sum(r.visible_count for r in results)
    total_estimated = sum(r.estimated_total for r in results)
    total_inferred = sum(r.occlusion_inferred for r in results)
    total_confirmed = sum(r.confirmed_count for r in results)
    total_confirmed_by_context = sum(r.confirmed_by_context_count for r in results)
    total_unknown = sum(r.unknown_count for r in results)

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
                "learned_unit_depth_m": r.learned_unit_depth_m,
                "method": r.method,
                "countability": r.countability,
                "countability_reasons": r.countability_reasons,
                "confirmed_count": r.confirmed_count,
                "confirmed_by_context_count": r.confirmed_by_context_count,
                "unknown_count": r.unknown_count,
                "dominant_class_id": r.dominant_class_id,
                "dominant_class_name": r.dominant_class_name,
            }
        )

    # Per-class aggregation
    per_class_summary: dict[str, dict[str, Any]] = {}
    for r in results:
        cls = r.class_name
        if cls not in per_class_summary:
            per_class_summary[cls] = {
                "countable_visible": 0,
                "uncountable_visible": 0,
                "uncountable_estimated": 0,
                "total": 0,
            }
        if r.countability == "countable":
            per_class_summary[cls]["countable_visible"] += r.visible_count
        else:
            per_class_summary[cls]["uncountable_visible"] += r.visible_count
            per_class_summary[cls]["uncountable_estimated"] += r.estimated_total
        per_class_summary[cls]["total"] += r.estimated_total

    return {
        "total_visible": total_visible,
        "total_estimated": total_estimated,
        "total_occlusion_inferred": total_inferred,
        "confirmed_count": total_confirmed,
        "confirmed_by_context_count": total_confirmed_by_context,
        "unknown_count": total_unknown,
        "filtered_count": filtered_count,
        "clusters": cluster_list,
        "per_class_summary": per_class_summary,
    }
