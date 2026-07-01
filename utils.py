"""Utility functions for the occlusion counting pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_image(path: Path | str) -> np.ndarray:
    """Load an image as BGR (OpenCV default)."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return image


def save_image(path: Path | str, image: np.ndarray) -> None:
    """Save an image, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ext = ".jpg" if suffix == ".jpeg" else suffix
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(path)


def load_class_names(data_yaml: Path | str) -> list[str]:
    """Load class names from a YOLO-style data.yaml."""
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        ordered = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
        return [v for _, v in ordered]
    raise ValueError(f"Invalid names format in {data_yaml}")


def build_data_yaml_seg(data_root: Path, class_names: list[str]) -> Path:
    """Create a YOLO-seg data.yaml if it does not exist.

    The segmentation label files are expected under {data_root}/labels/train
    and {data_root}/labels/val as polygon-format .txt files.
    """
    data_root = Path(data_root)
    yaml_path = data_root / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = (
        f"path: {data_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resize_keep_aspect(
    image: np.ndarray,
    target_size: int,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Resize image with aspect-ratio preserving padding.

    Returns:
        resized_image, (scale_x, scale_y), (pad_left, pad_top)
    """
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded, (scale, scale), (pad_left, pad_top)


def restore_coords_after_padding(
    coords: np.ndarray,
    pad_left: int,
    pad_top: int,
    scale: float,
) -> np.ndarray:
    """Restore coordinates from padded/resized image back to original image space."""
    coords = coords.copy()
    coords[..., 0] = (coords[..., 0] - pad_left) / scale
    coords[..., 1] = (coords[..., 1] - pad_top) / scale
    return coords


def save_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
