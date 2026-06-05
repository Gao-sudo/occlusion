"""Mask geometry analysis for occlusion counting.

Provides clustering of instance masks along hooks/stacks and 1-D projection
of masks along their principal axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from occlusion.config import CLUSTER_EPS_PX, CLUSTER_MIN_SAMPLES


@dataclass
class MaskInfo:
    """Info for a single instance mask."""
    mask: np.ndarray          # bool HxW
    class_id: int
    class_name: str
    confidence: float
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


@dataclass
class ClusterInfo:
    """A group of masks belonging to the same hook / stack."""
    cluster_id: int
    masks: list[MaskInfo]
    # axis direction (dx, dy) normalized, pointing along the stack/hook
    axis_direction: tuple[float, float]
    # approximate center of the cluster
    center: tuple[float, float]


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
