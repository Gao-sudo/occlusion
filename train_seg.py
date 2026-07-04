"""Training script for YOLO-seg instance segmentation model.

This script is completely independent from train/*.py and does not import
any existing training modules, preserving the "no intrusion" requirement.

Example:
    python -m occlusion.train_seg \
        --data-root ./data_occlusion \
        --epochs 200 \
        --device 0
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from occlusion.config import (
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_IMGSZ,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRETRAINED_SEG,
    PROJECT_ROOT,
)
from occlusion.utils import build_data_yaml_seg, ensure_dir, load_class_names, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO-seg for occlusion counting")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data_occlusion")
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--weights", type=str, default=str(DEFAULT_PRETRAINED_SEG))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mixup", type=float, default=0.15)
    parser.add_argument("--project", type=str, default="runs/occlusion_seg")
    parser.add_argument("--name", type=str, default="yolov11s_seg_jomoo")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "occlusion")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    if args.data_yaml is None:
        class_names = load_class_names(PROJECT_ROOT / "data.yaml")
        data_yaml = build_data_yaml_seg(data_root, class_names)
    else:
        data_yaml = Path(args.data_yaml)

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ensure_dir(args.output_root)

    print("=" * 60)
    print("Occlusion Counting - YOLO-seg Training")
    print(f"Data: {data_yaml}")
    print(f"Weights: {args.weights}")
    print(f"Output: {out_root}")
    print("=" * 60)

    os.environ["YOLO_VERBOSE"] = "False"
    model = YOLO(args.weights)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cache=False,
        pretrained=True,
        amp=True,
        verbose=False,
        plots=True,
    )

    # Organize artifacts similar to existing outputs layout
    run_dir = Path(model.trainer.save_dir)
    artifact_dir = ensure_dir(out_root / tag)
    weights_dir = ensure_dir(artifact_dir / "weights")
    visual_dir = ensure_dir(artifact_dir / "visualizations")
    logs_dir = ensure_dir(artifact_dir / "logs")
    meta_dir = ensure_dir(artifact_dir / "meta")

    for src_name in ("best.pt", "last.pt"):
        src = run_dir / "weights" / src_name
        if src.exists():
            import shutil
            shutil.copy2(src, weights_dir / src_name)

    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for f in run_dir.glob(pattern):
            import shutil
            shutil.copy2(f, visual_dir / f.name)

    for src_name in ("results.csv", "args.yaml"):
        src = run_dir / src_name
        if src.exists():
            import shutil
            shutil.copy2(src, logs_dir / src_name)

    meta = {
        "model_name": "yolov11s_seg",
        "weights_source": args.weights,
        "data_yaml": str(data_yaml),
        "training_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta_dir / "summary.json", meta)

    print("=" * 60)
    print("Training complete.")
    print(f"Artifacts: {artifact_dir}")
    print(f"Best weights: {weights_dir / 'best.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
