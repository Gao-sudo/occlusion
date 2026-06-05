"""Visualization helpers for occlusion counting results."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from occlusion.config import DEFAULT_FONT_SCALE, DEFAULT_LINE_WIDTH
from occlusion.fusion_counter import CountResult
from occlusion.mask_analyzer import ClusterInfo, MaskInfo


def _palette(n: int) -> list[tuple[int, int, int]]:
    """Generate a BGR color palette."""
    colors = [
        (32, 32, 220),    # red
        (32, 170, 32),    # green
        (40, 120, 235),   # orange
        (180, 60, 180),   # purple
        (20, 190, 190),   # yellow-ish
        (90, 40, 200),    # magenta
        (0, 165, 255),    # cyan-ish
        (128, 128, 0),    # dark cyan
    ]
    return [colors[i % len(colors)] for i in range(n)]


def draw_masks(
    image: np.ndarray,
    clusters: list[ClusterInfo],
    alpha: float = 0.4,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> np.ndarray:
    """Overlay instance masks with cluster-specific colors."""
    vis = image.copy()
    colors = _palette(len(clusters))
    overlay = vis.copy()

    for cidx, cluster in enumerate(clusters):
        color = colors[cidx]
        for m in cluster.masks:
            overlay[m.mask] = color
            contours, _ = cv2.findContours(
                m.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, line_width)

    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis


def draw_cluster_info(
    image: np.ndarray,
    clusters: list[ClusterInfo],
    results: list[CountResult],
    font_scale: float = DEFAULT_FONT_SCALE,
) -> np.ndarray:
    """Draw cluster labels and count results on the image."""
    vis = image.copy()
    colors = _palette(len(clusters))

    for cidx, (cluster, result) in enumerate(zip(clusters, results)):
        color = colors[cidx]
        cx, cy = cluster.center
        cx, cy = int(cx), int(cy)

        label = (
            f"C{cidx} {result.class_name}: "
            f"vis={result.visible_count} est={result.estimated_total} "
            f"({result.confidence})"
        )
        if result.occlusion_inferred > 0:
            label += f" +{result.occlusion_inferred} occluded"

        # Draw background rectangle for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (cx, cy - th - 6), (cx + tw + 4, cy + 4), color, -1)
        cv2.putText(
            vis,
            label,
            (cx + 2, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw principal axis
        if cluster.masks:
            ax, ay = cluster.axis_direction
            line_len = 80
            x1 = int(cx - ax * line_len)
            y1 = int(cy - ay * line_len)
            x2 = int(cx + ax * line_len)
            y2 = int(cy + ay * line_len)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)

    return vis


def draw_depth_profile(
    image: np.ndarray,
    result: CountResult,
    cluster_color: tuple[int, int, int] = (0, 255, 0),
    plot_height: int = 120,
) -> np.ndarray:
    """Draw a small depth-profile strip for a single cluster onto the image."""
    if len(result.depth_values) == 0:
        return image

    vis = image.copy()
    h, w = vis.shape[:2]

    # Normalize depth to 0-1 for visualization
    dmin, dmax = result.depth_values.min(), result.depth_values.max()
    norm = (result.depth_values - dmin) / (dmax - dmin + 1e-9)

    # Draw strip at bottom
    strip_y0 = h - plot_height - 10
    strip_w = min(w, len(norm))
    strip = np.zeros((plot_height, strip_w, 3), dtype=np.uint8)

    for i, v in enumerate(norm[:strip_w]):
        y = int((1.0 - v) * (plot_height - 1))
        cv2.line(strip, (i, plot_height - 1), (i, y), cluster_color, 1)

    # Draw step boundaries
    for s, e in result.depth_steps:
        # map position to pixel in strip
        if len(result.depth_positions) == 0:
            continue
        pmin, pmax = result.depth_positions.min(), result.depth_positions.max()
        rng = pmax - pmin + 1e-9
        xs = int((s - pmin) / rng * strip_w)
        xe = int((e - pmin) / rng * strip_w)
        cv2.line(strip, (xs, 0), (xs, plot_height - 1), (0, 0, 255), 1)
        cv2.line(strip, (xe, 0), (xe, plot_height - 1), (0, 0, 255), 1)

    vis[strip_y0 : strip_y0 + plot_height, 0:strip_w] = strip
    cv2.putText(
        vis,
        f"Cluster {result.cluster_id} depth profile",
        (4, strip_y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        cluster_color,
        1,
        cv2.LINE_AA,
    )
    return vis


def compose_result_image(
    original_bgr: np.ndarray,
    depth_map: np.ndarray,
    clusters: list[ClusterInfo],
    results: list[CountResult],
) -> np.ndarray:
    """Compose a side-by-side visualization: original+masks | depth heatmap."""
    h, w = original_bgr.shape[:2]

    # Left: original with masks and labels
    left = draw_masks(original_bgr, clusters)
    left = draw_cluster_info(left, clusters, results)

    # Right: depth colormap
    dmin, dmax = depth_map.min(), depth_map.max()
    depth_norm = ((depth_map - dmin) / (dmax - dmin + 1e-9) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Also overlay masks on depth map for reference
    right = draw_masks(depth_color, clusters, alpha=0.3)
    right = draw_cluster_info(right, clusters, results)

    # Concatenate horizontally
    canvas = np.concatenate([left, right], axis=1)

    # Add summary text at top
    total_est = sum(r.estimated_total for r in results)
    total_vis = sum(r.visible_count for r in results)
    summary = f"Visible: {total_vis} | Estimated total: {total_est}"
    cv2.putText(
        canvas,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas
