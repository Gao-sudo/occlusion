"""Helpers for converting labels and polygons for YOLO-seg workflows."""
from __future__ import annotations

from typing import Iterable, Any

import cv2
import numpy as np


def parse_yolo_bbox_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO bbox row.

    Returns:
        (class_id, x_center, y_center, width, height) in normalized coords.
        Returns None for blank lines.
    """
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields, got {len(parts)}: {line!r}")

    class_id = int(parts[0])
    x_center, y_center, width, height = (float(v) for v in parts[1:])
    return class_id, x_center, y_center, width, height


def yolo_bbox_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    min_size: float = 1e-6,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO bbox values into normalized XYXY coordinates."""
    width = max(float(width), min_size)
    height = max(float(height), min_size)

    x1 = np.clip(x_center - width / 2.0, 0.0, 1.0)
    y1 = np.clip(y_center - height / 2.0, 0.0, 1.0)
    x2 = np.clip(x_center + width / 2.0, 0.0, 1.0)
    y2 = np.clip(y_center + height / 2.0, 0.0, 1.0)

    if x2 <= x1:
        x2 = min(1.0, x1 + min_size)
    if y2 <= y1:
        y2 = min(1.0, y1 + min_size)
    return x1, y1, x2, y2


def scale_bbox_xyxy(
    bbox_xyxy_norm: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Scale normalized XYXY bbox coordinates into pixel-space integers."""
    height, width = image_shape
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    x1n, y1n, x2n, y2n = bbox_xyxy_norm
    x1 = int(np.clip(round(x1n * width), 0, max(0, width - 1)))
    y1 = int(np.clip(round(y1n * height), 0, max(0, height - 1)))
    x2 = int(np.clip(round(x2n * width), 0, max(0, width - 1)))
    y2 = int(np.clip(round(y2n * height), 0, max(0, height - 1)))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def yolo_bbox_to_polygon(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    min_size: float = 1e-6,
) -> np.ndarray:
    """Convert normalized YOLO bbox values into a 4-point polygon."""
    x1, y1, x2, y2 = yolo_bbox_to_xyxy(x_center, y_center, width, height, min_size=min_size)
    polygon = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    return normalize_polygon_points(polygon)


def bbox_xyxy_to_polygon(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Convert pixel-space XYXY bbox coordinates into a 4-point polygon."""
    return np.array(
        [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ],
        dtype=np.float32,
    )


def normalize_polygon_points(
    points: np.ndarray | Iterable[Iterable[float]],
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normalize polygon points to [0, 1] if image shape is provided, then clip."""
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if polygon.size == 0:
        return polygon

    if image_shape is not None:
        height, width = image_shape
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image shape: {image_shape}")
        polygon[:, 0] /= float(width)
        polygon[:, 1] /= float(height)

    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, 1.0)
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, 1.0)
    return polygon


def polygon_to_yolo_seg_line(class_id: int, polygon_xy_norm: np.ndarray) -> str:
    """Serialize a normalized polygon into a YOLO-seg label line."""
    polygon = normalize_polygon_points(polygon_xy_norm)
    if len(polygon) < 3:
        raise ValueError("Polygon must contain at least 3 points")
    coords = " ".join(f"{float(v):.6f}" for v in polygon.reshape(-1))
    return f"{int(class_id)} {coords}"


def shrink_box(x1: int, y1: int, x2: int, y2: int, ratio: float = 0.25) -> tuple[int, int, int, int]:
    """Shrink a bbox inward by a ratio on both axes."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    dx = int(round(w * ratio))
    dy = int(round(h * ratio))
    xs1 = min(x2 - 1, x1 + dx)
    ys1 = min(y2 - 1, y1 + dy)
    xs2 = max(xs1 + 1, x2 - dx)
    ys2 = max(ys1 + 1, y2 - dy)
    return xs1, ys1, xs2, ys2


def sample_points_in_box(x1: int, y1: int, x2: int, y2: int, k: int = 3) -> np.ndarray:
    """Sample a few stable positive points inside a bbox."""
    xs = np.linspace(x1, x2, num=k + 2, dtype=int)[1:-1]
    ys = np.linspace(y1, y2, num=k + 2, dtype=int)[1:-1]
    points: list[list[int]] = []
    points.append([(x1 + x2) // 2, (y1 + y2) // 2])
    if k >= 2 and len(xs) > 0 and len(ys) > 0:
        points.append([int(xs[0]), int(ys[0])])
    if k >= 3 and len(xs) > 0 and len(ys) > 0:
        points.append([int(xs[-1]), int(ys[-1])])
    return np.asarray(points, dtype=np.int32)


def ring_negative_points(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    inset: int = 3,
    n_each_side: int = 3,
) -> np.ndarray:
    """Sample a ring of negative points near the inner bbox border."""
    x1i, y1i, x2i, y2i = x1 + inset, y1 + inset, x2 - inset, y2 - inset
    if x2i <= x1i or y2i <= y1i:
        return np.zeros((0, 2), dtype=np.int32)

    xs = np.linspace(x1i, x2i, n_each_side + 2, dtype=int)[1:-1]
    ys = np.linspace(y1i, y2i, n_each_side + 2, dtype=int)[1:-1]

    points: list[list[int]] = []
    for x in xs:
        points.append([int(x), int(y1i)])
        points.append([int(x), int(y2i)])
    for y in ys:
        points.append([int(x1i), int(y)])
        points.append([int(x2i), int(y)])
    return np.asarray(points, dtype=np.int32)


def largest_component_containing_center(mask01: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Keep the connected component containing the box center, or the largest one."""
    num, labels = cv2.connectedComponents(mask01.astype(np.uint8))
    if num <= 1:
        return mask01.astype(np.uint8)

    h, w = mask01.shape
    cx = int(np.clip(cx, 0, max(0, w - 1)))
    cy = int(np.clip(cy, 0, max(0, h - 1)))
    center_label = labels[cy, cx]
    if center_label != 0:
        return (labels == center_label).astype(np.uint8)

    areas = [(lab, int((labels == lab).sum())) for lab in range(1, num)]
    if not areas:
        return mask01.astype(np.uint8)
    lab_max = max(areas, key=lambda x: x[1])[0]
    return (labels == lab_max).astype(np.uint8)


def score_sam_mask(
    mask01: np.ndarray,
    bbox_area: float,
    prefer_no_miss: bool = True,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 1.15,
) -> float:
    """Score a SAM candidate mask for single-product extraction."""
    area = float(mask01.sum())
    ratio = area / max(1.0, bbox_area)

    if ratio < min_area_ratio:
        return -1e9
    if ratio > max_area_ratio:
        return -1e9

    num_cc, _ = cv2.connectedComponents(mask01.astype(np.uint8))
    cc_penalty = max(0, num_cc - 1) * 0.15

    target = 0.85 if prefer_no_miss else 0.65
    score = 1.0 - abs(ratio - target)
    return score - cc_penalty


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.01,
    min_points: int = 4,
    min_contour_area: float = 0.0,
    chain_method: int = cv2.CHAIN_APPROX_SIMPLE,
) -> np.ndarray | None:
    """Extract a stable polygon from a binary mask.

    The largest external contour is simplified with approxPolyDP.
    Returns pixel-space polygon points, or None if extraction fails.
    """
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or mask_u8.size == 0 or not mask_u8.any():
        return None

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, chain_method)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < float(min_contour_area):
        return None

    perimeter = cv2.arcLength(contour, True)
    epsilon = max(perimeter * float(epsilon_ratio), 1.0)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    polygon = approx.reshape(-1, 2).astype(np.float32)

    if len(polygon) < min_points:
        rect = cv2.boxPoints(cv2.minAreaRect(contour))
        polygon = rect.astype(np.float32)

    if len(polygon) < min_points:
        return None
    return polygon


def sam_mask_to_polygon(
    mask01: np.ndarray,
    epsilon_ratio: float = 0.0045,
    min_area_px: int = 30,
) -> np.ndarray | None:
    """Convert a cleaned SAM mask into a simplified pixel-space polygon."""
    mask_u8 = (mask01 > 0).astype(np.uint8) * 255
    return mask_to_polygon(
        mask_u8,
        epsilon_ratio=epsilon_ratio,
        min_points=3,
        min_contour_area=float(min_area_px),
        chain_method=cv2.CHAIN_APPROX_TC89_KCOS,
    )


def refine_bbox_to_polygon_with_sam(
    predictor: Any,
    bbox_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    expand_ratio: float = 0.02,
    shrink_ratio: float = 0.28,
    point_samples: int = 3,
    negative_inset: int = 3,
    negative_points_per_side: int = 2,
    prefer_no_miss: bool = True,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 1.15,
    min_mask_area_px: int = 30,
    epsilon_ratio: float = 0.0045,
    fallback_mode: str = "skip",
) -> tuple[np.ndarray | None, str, dict[str, float | int | bool]]:
    """Refine one bbox into a polygon using a preloaded SAM predictor."""
    h, w = image_shape
    x1, y1, x2, y2 = map(int, bbox_xyxy)

    pad_w = int(round(max(1, x2 - x1) * expand_ratio))
    pad_h = int(round(max(1, y2 - y1) * expand_ratio))
    x1e = max(0, x1 - pad_w)
    y1e = max(0, y1 - pad_h)
    x2e = min(w - 1, x2 + pad_w)
    y2e = min(h - 1, y2 + pad_h)

    xi1, yi1, xi2, yi2 = shrink_box(x1e, y1e, x2e, y2e, ratio=shrink_ratio)
    pos_pts = sample_points_in_box(xi1, yi1, xi2, yi2, k=point_samples)
    neg_pts = ring_negative_points(x1e, y1e, x2e, y2e, inset=negative_inset, n_each_side=negative_points_per_side)

    if len(neg_pts) > 0:
        points = np.concatenate([pos_pts, neg_pts], axis=0).astype(np.float32)
        labels = np.concatenate([np.ones(len(pos_pts)), np.zeros(len(neg_pts))], axis=0).astype(np.int32)
    else:
        points = pos_pts.astype(np.float32)
        labels = np.ones(len(pos_pts), dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.array([x1e, y1e, x2e, y2e], dtype=np.float32),
        multimask_output=True,
    )

    bbox_area = float(max(1, x2e - x1e) * max(1, y2e - y1e))
    best_idx = -1
    best_score = -1e18

    for idx in range(int(masks.shape[0])):
        mask01 = masks[idx].astype(np.uint8)
        candidate_score = score_sam_mask(
            mask01,
            bbox_area=bbox_area,
            prefer_no_miss=prefer_no_miss,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        candidate_score += float(scores[idx]) * 0.15
        if candidate_score > best_score:
            best_score = candidate_score
            best_idx = idx

    meta: dict[str, float | int | bool] = {
        "bbox_area": bbox_area,
        "best_score": float(best_score),
        "accepted": False,
    }

    polygon_norm: np.ndarray | None = None
    source = "rejected"

    if best_idx >= 0 and best_score > -1e8:
        mask01 = masks[best_idx].astype(np.uint8)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        mask01 = largest_component_containing_center(mask01, cx, cy)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask01 = cv2.morphologyEx(mask01, cv2.MORPH_CLOSE, kernel, iterations=1)
        polygon_px = sam_mask_to_polygon(mask01, epsilon_ratio=epsilon_ratio, min_area_px=min_mask_area_px)
        if polygon_px is not None:
            polygon_norm = normalize_polygon_points(polygon_px, image_shape=image_shape)
            source = "sam"
            meta["accepted"] = True

    if polygon_norm is None and fallback_mode == "bbox":
        polygon_norm = normalize_polygon_points(bbox_xyxy_to_polygon(x1e, y1e, x2e, y2e), image_shape=image_shape)
        source = "bbox_fallback"

    return polygon_norm, source, meta


def build_detection_polygon(
    mask: np.ndarray | None,
    bbox_xyxy: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    epsilon_ratio: float = 0.01,
) -> tuple[np.ndarray, str]:
    """Build a normalized polygon for one detection.

    Prefers a polygon extracted from the predicted mask. Falls back to the
    detection bbox rectangle when the mask is empty or unstable.
    """
    if mask is not None:
        polygon_px = mask_to_polygon(mask, epsilon_ratio=epsilon_ratio)
        if polygon_px is not None:
            return normalize_polygon_points(polygon_px, image_shape=image_shape), "mask"

    x1, y1, x2, y2 = bbox_xyxy
    bbox_polygon = bbox_xyxy_to_polygon(x1, y1, x2, y2)
    return normalize_polygon_points(bbox_polygon, image_shape=image_shape), "bbox_fallback"


def polygon_to_points_list(polygon: np.ndarray) -> list[list[float]]:
    """Convert polygon points into plain JSON-friendly lists."""
    return [[float(x), float(y)] for x, y in np.asarray(polygon, dtype=np.float32).reshape(-1, 2)]
