"""Prepare a YOLO-seg dataset from YOLO bbox labels."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from occlusion.config import DEFAULT_DATA_ROOT, PROJECT_ROOT
from occlusion.label_convert import (
    parse_yolo_bbox_line,
    polygon_to_yolo_seg_line,
    refine_bbox_to_polygon_with_sam,
    scale_bbox_xyxy,
    yolo_bbox_to_polygon,
    yolo_bbox_to_xyxy,
)
from occlusion.utils import build_data_yaml_seg, ensure_dir, load_class_names, load_image, save_image


DEFAULT_ARCHIVE_ROOT = Path(
    "C:/Users/China/Documents/xwechat_files/wxid_r5wqao29yq8e21_c1f8/msg/file/2026-06/归档"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import YOLO bbox dataset and convert labels to YOLO-seg polygons")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in destination")
    parser.add_argument("--dry-run", action="store_true", help="Report planned work without writing files")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on missing pairs or malformed labels")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to import",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only convert the first N images per selected split; useful for previews",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for visualization overlays of converted polygons",
    )
    parser.add_argument(
        "--polygon-mode",
        choices=["bbox", "sam"],
        default="bbox",
        help="How to generate training polygons from bbox labels",
    )
    parser.add_argument("--sam-checkpoint", type=Path, default=None, help="Path to the SAM checkpoint file")
    parser.add_argument("--sam-model-type", type=str, default="vit_l", help="SAM model type: vit_b, vit_l, vit_h")
    parser.add_argument("--sam-device", type=str, default="cuda", help="Device for SAM inference, e.g. cuda or cpu")
    parser.add_argument(
        "--sam-fallback-mode",
        choices=["skip", "bbox"],
        default="skip",
        help="Behavior when SAM refinement fails",
    )
    parser.add_argument("--sam-expand-ratio", type=float, default=0.02)
    parser.add_argument("--sam-shrink-ratio", type=float, default=0.28)
    parser.add_argument("--sam-min-mask-area", type=int, default=30)
    parser.add_argument("--sam-min-area-ratio", type=float, default=0.15)
    parser.add_argument("--sam-max-area-ratio", type=float, default=1.15)
    parser.add_argument("--sam-epsilon-ratio", type=float, default=0.0045)
    return parser.parse_args()


def load_sam_predictor(checkpoint: Path, model_type: str, device: str) -> Any:
    from segment_anything import SamPredictor, sam_model_registry
    import torch

    resolved_device = device
    if device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=resolved_device)
    predictor = SamPredictor(sam)
    predictor._occlusion_device = resolved_device  # type: ignore[attr-defined]
    return predictor


def iter_image_files(images_dir: Path) -> list[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def convert_label_lines(
    lines: list[str],
    image_shape: tuple[int, int],
    polygon_mode: str,
    predictor: Any | None,
    sam_options: dict[str, Any],
) -> tuple[list[str], list[dict[str, object]]]:
    converted: list[str] = []
    objects: list[dict[str, object]] = []

    for line_number, line in enumerate(lines, start=1):
        parsed = parse_yolo_bbox_line(line)
        if parsed is None:
            continue
        class_id, x_center, y_center, width, height = parsed
        bbox_xyxy_norm = yolo_bbox_to_xyxy(x_center, y_center, width, height)
        bbox_xyxy_px = scale_bbox_xyxy(bbox_xyxy_norm, image_shape)
        bbox_polygon_norm = yolo_bbox_to_polygon(x_center, y_center, width, height)

        polygon_norm = bbox_polygon_norm
        source = "bbox"
        meta: dict[str, object] = {}

        if polygon_mode == "sam":
            if predictor is None:
                raise RuntimeError("SAM mode requested but predictor was not initialized")
            refined_polygon, source, meta = refine_bbox_to_polygon_with_sam(
                predictor,
                bbox_xyxy_px,
                image_shape,
                expand_ratio=float(sam_options["expand_ratio"]),
                shrink_ratio=float(sam_options["shrink_ratio"]),
                min_mask_area_px=int(sam_options["min_mask_area"]),
                min_area_ratio=float(sam_options["min_area_ratio"]),
                max_area_ratio=float(sam_options["max_area_ratio"]),
                epsilon_ratio=float(sam_options["epsilon_ratio"]),
                fallback_mode=str(sam_options["fallback_mode"]),
            )
            if refined_polygon is None:
                source = "rejected"
            else:
                polygon_norm = refined_polygon

        if source != "rejected":
            converted.append(polygon_to_yolo_seg_line(class_id, polygon_norm))

        objects.append(
            {
                "line_number": line_number,
                "class_id": class_id,
                "bbox_norm": np.asarray(bbox_xyxy_norm, dtype=np.float32),
                "bbox_xyxy_px": bbox_xyxy_px,
                "polygon_norm": polygon_norm,
                "source": source,
                "meta": meta,
            }
        )

    return converted, objects


def convert_label_file(
    src_label: Path,
    dst_label: Path,
    image_shape: tuple[int, int],
    polygon_mode: str,
    predictor: Any | None,
    sam_options: dict[str, Any],
    dry_run: bool,
) -> tuple[int, int, list[dict[str, object]]]:
    lines = src_label.read_text(encoding="utf-8").splitlines()
    converted, objects = convert_label_lines(lines, image_shape, polygon_mode, predictor, sam_options)

    if not dry_run:
        dst_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    accepted_count = sum(1 for obj in objects if obj["source"] != "rejected")
    return len(lines), accepted_count, objects


def copy_image(src_image: Path, dst_image: Path, overwrite: bool, dry_run: bool) -> bool:
    if dst_image.exists() and not overwrite:
        return False
    if not dry_run:
        shutil.copy2(src_image, dst_image)
    return True


def _polygon_to_contour(polygon_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    polygon_px = np.asarray(polygon_norm, dtype=np.float32).copy().reshape(-1, 2)
    polygon_px[:, 0] *= float(width)
    polygon_px[:, 1] *= float(height)
    return np.round(polygon_px).astype(np.int32).reshape(-1, 1, 2)


def draw_preview(image_path: Path, objects: list[dict[str, object]], preview_path: Path) -> None:
    image = load_image(image_path)
    height, width = image.shape[:2]

    for obj in objects:
        x1, y1, x2, y2 = (int(v) for v in obj["bbox_xyxy_px"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)

        source = str(obj["source"])
        color_map = {
            "sam": (0, 220, 0),
            "bbox": (0, 165, 255),
            "bbox_fallback": (0, 165, 255),
            "rejected": (0, 0, 255),
        }
        color = color_map.get(source, (255, 255, 255))

        polygon_norm = np.asarray(obj["polygon_norm"], dtype=np.float32)
        if source != "rejected" and polygon_norm.size > 0:
            contour = _polygon_to_contour(polygon_norm, width, height)
            cv2.polylines(image, [contour], True, color, 2)
            anchor = contour[0, 0]
            text_origin = (int(anchor[0]), max(18, int(anchor[1]) - 4))
        else:
            text_origin = (x1, max(18, y1 - 4))

        label = f"{obj['class_id']}:{source}"
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    save_image(preview_path, image)


def prepare_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    overwrite: bool,
    dry_run: bool,
    strict: bool,
    max_images: int | None,
    preview_dir: Path | None,
    polygon_mode: str,
    predictor: Any | None,
    sam_options: dict[str, Any],
) -> dict[str, int]:
    images_dir = src_root / "images" / split
    labels_dir = src_root / "labels" / split
    dst_images_dir = ensure_dir(dst_root / "images" / split)
    dst_labels_dir = ensure_dir(dst_root / "labels" / split)
    preview_split_dir = ensure_dir(preview_dir / split) if preview_dir is not None else None

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing split directories for {split}: {images_dir} / {labels_dir}")

    stats = {
        "images_seen": 0,
        "images_copied": 0,
        "labels_written": 0,
        "objects_written": 0,
        "skipped_existing": 0,
        "preview_written": 0,
        "objects_attempted": 0,
        "sam_accepted": 0,
        "sam_rejected": 0,
        "bbox_fallback_count": 0,
    }

    image_files = iter_image_files(images_dir)
    if max_images is not None:
        image_files = image_files[:max_images]

    for image_path in image_files:
        stats["images_seen"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            message = f"Missing label for image: {image_path.name}"
            if strict:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}")
            continue

        image = load_image(image_path)
        if polygon_mode == "sam" and predictor is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

        dst_image = dst_images_dir / image_path.name
        dst_label = dst_labels_dir / f"{image_path.stem}.txt"

        try:
            raw_lines, object_count, objects = convert_label_file(
                label_path,
                dst_label,
                image.shape[:2],
                polygon_mode,
                predictor,
                sam_options,
                dry_run=dry_run,
            )
        except Exception as exc:
            message = f"Failed to convert {label_path.name}: {exc}"
            if strict:
                raise ValueError(message) from exc
            print(f"[WARN] {message}")
            continue

        copied = copy_image(image_path, dst_image, overwrite=overwrite, dry_run=dry_run)
        if copied:
            stats["images_copied"] += 1
        else:
            stats["skipped_existing"] += 1

        stats["labels_written"] += 1
        stats["objects_written"] += object_count
        stats["objects_attempted"] += len(objects)
        stats["sam_accepted"] += sum(1 for obj in objects if obj["source"] == "sam")
        stats["sam_rejected"] += sum(1 for obj in objects if obj["source"] == "rejected")
        stats["bbox_fallback_count"] += sum(1 for obj in objects if obj["source"] in {"bbox", "bbox_fallback"})

        if preview_split_dir is not None and not dry_run:
            preview_path = preview_split_dir / image_path.name
            draw_preview(image_path, objects, preview_path)
            stats["preview_written"] += 1

        if raw_lines == 0:
            print(f"[INFO] Empty label file kept: {label_path.name}")

    return stats


def main() -> None:
    args = parse_args()
    class_names = load_class_names(PROJECT_ROOT / "data.yaml")
    preview_dir = ensure_dir(args.preview_dir) if args.preview_dir is not None and not args.dry_run else args.preview_dir

    predictor = None
    if args.polygon_mode == "sam":
        if args.sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required when --polygon-mode sam")
        predictor = load_sam_predictor(args.sam_checkpoint, args.sam_model_type, args.sam_device)

    sam_options = {
        "expand_ratio": args.sam_expand_ratio,
        "shrink_ratio": args.sam_shrink_ratio,
        "min_mask_area": args.sam_min_mask_area,
        "min_area_ratio": args.sam_min_area_ratio,
        "max_area_ratio": args.sam_max_area_ratio,
        "epsilon_ratio": args.sam_epsilon_ratio,
        "fallback_mode": args.sam_fallback_mode,
    }

    all_stats: dict[str, dict[str, int]] = {}
    for split in args.splits:
        stats = prepare_split(
            src_root=args.src_root,
            dst_root=args.dst_root,
            split=split,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            strict=args.strict,
            max_images=args.max_images,
            preview_dir=preview_dir,
            polygon_mode=args.polygon_mode,
            predictor=predictor,
            sam_options=sam_options,
        )
        all_stats[split] = stats

    if not args.dry_run:
        build_data_yaml_seg(args.dst_root, class_names)

    print("=" * 60)
    print("Dataset preparation complete")
    print(f"Source: {args.src_root}")
    print(f"Destination: {args.dst_root}")
    print(f"Dry run: {args.dry_run}")
    print(f"Polygon mode: {args.polygon_mode}")
    if preview_dir is not None:
        print(f"Preview dir: {preview_dir}")
    for split, stats in all_stats.items():
        print(
            f"[{split}] images_seen={stats['images_seen']} images_copied={stats['images_copied']} "
            f"labels_written={stats['labels_written']} objects_written={stats['objects_written']} "
            f"skipped_existing={stats['skipped_existing']} preview_written={stats['preview_written']} "
            f"objects_attempted={stats['objects_attempted']} sam_accepted={stats['sam_accepted']} "
            f"sam_rejected={stats['sam_rejected']} bbox_fallback_count={stats['bbox_fallback_count']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
