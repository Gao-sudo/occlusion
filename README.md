# Occlusion Counting

JOMOO product instance-segmentation and occlusion-aware counting pipeline.

The project trains a YOLO segmentation model for hanging retail products, runs
single-image or batch inference, filters common display-board false positives,
and returns both counting summaries and polygon-level instance metadata.

## Features

- YOLO-seg training for custom JOMOO product classes.
- Image and directory inference from the command line.
- FastAPI service for count and analyze endpoints.
- Chinese class-name output from YOLO-style `data.yaml`.
- Polygon, bounding box, confidence, mask area, centroid, and orientation metadata.
- Heuristic filtering for large horizontal display signs above hanging products.
- Optional Depth Anything V2 based occlusion counting, with `--skip-depth` fallback.

## Project Layout

```text
occlusion/
  api.py                  FastAPI service
  config.py               Default paths and inference/training settings
  depth_estimator.py      Depth Anything V2 wrapper
  fusion_counter.py       Visible and occlusion-inferred count fusion
  infer.py                CLI inference pipeline
  label_convert.py        BBox, polygon, and mask conversion helpers
  mask_analyzer.py        Mask geometry, clustering, and false-positive filters
  prepare_seg_dataset.py  Convert YOLO bbox datasets to YOLO-seg labels
  train_seg.py            YOLO-seg training entrypoint
  utils.py                Image, YAML, JSON, and filesystem helpers
  visualizer.py           Mask, label, and count visualization
```

Data, model weights, runs, and output artifacts are intentionally ignored by
Git. Keep them outside version control or provide them separately.

## Class Names

The default dataset expects these classes:

```yaml
names:
  0: 九牧增压花洒
  1: 九牧增压花洒套装
  2: 九牧大冲力喷枪角阀
  3: 九牧安全快开
  4: 九牧安全角阀
  5: 九牧百搭下水
  6: 九牧百搭下水（软袋）
  7: 九牧轻音盖板
  8: 九牧防断裂淋浴软管
  9: 九牧防漏水件
  10: 九牧防爆编织软管
  11: 九牧防臭下水管
  12: 九牧防臭地漏
  13: 九牧健康编织软管
```

## Environment

Python 3.8+ is recommended. Install the runtime dependencies used by the
pipeline:

```bash
pip install ultralytics opencv-python numpy pyyaml scikit-learn pillow fastapi uvicorn python-multipart
```

Install PyTorch according to your CUDA or CPU environment before running
training or GPU inference.

## Dataset

The default YOLO-seg dataset layout is:

```text
data_occlusion/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

If the source labels are YOLO bounding boxes, convert them to segmentation
polygons first:

```bash
python -m occlusion.prepare_seg_dataset \
  --src-root /path/to/source_dataset \
  --dst-root /path/to/data_occlusion \
  --splits train val \
  --polygon-mode bbox
```

For SAM-assisted polygon refinement, use `--polygon-mode sam` and provide the
SAM checkpoint path.

## Training

Run from the parent directory that contains the `occlusion` package:

```bash
python -m occlusion.train_seg \
  --data-root ./data_occlusion \
  --epochs 200 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --name occlusion_seg
```

Training artifacts are copied to:

```text
outputs/occlusion/occlusion/<run_tag>/
  weights/best.pt
  weights/last.pt
  visualizations/
  logs/
  meta/summary.json
```

## Inference

Run inference on one image:

```bash
python -m occlusion.infer \
  --source /path/to/image.jpg \
  --weights /path/to/best.pt \
  --data-yaml /path/to/data_occlusion/data.yaml \
  --device 0 \
  --skip-depth
```

Run inference on a directory:

```bash
python -m occlusion.infer \
  --source /path/to/images \
  --weights /path/to/best.pt \
  --run-tag batch_test \
  --skip-depth
```

Outputs are saved under:

```text
outputs/occlusion/occlusion_infer/<run_tag>/
  visualizations/
  meta/results.json
```

`results.json` contains:

- `summary`: total visible count, estimated total, inferred occlusion count, and cluster summaries.
- `instances`: kept product detections used for clustering and counting.
- `filtered_instances`: detections removed by post-processing filters.

## Display Sign Filter

Some images contain a large horizontal sign above the product hooks. If YOLO
classifies that sign as a product, it is filtered before clustering and
counting.

The current filter is geometry-based and removes detections that are:

- large relative to the image,
- centered near the top of the image,
- close to horizontal by mask principal orientation.

Filtered detections are still written to `filtered_instances` with
`filter_reason: top_horizontal_display_sign` for debugging.

## API

Start the service:

```bash
uvicorn occlusion.api:app --host 0.0.0.0 --port 8001
```

Useful environment variables:

```bash
OCCLUSION_WEIGHTS=/path/to/best.pt
OCCLUSION_DEVICE=0
OCCLUSION_IMGSZ=640
OCCLUSION_CONF=0.25
OCCLUSION_IOU=0.5
OCCLUSION_SKIP_DEPTH=true
```

Endpoints:

- `POST /api/v1/occlusion/count`
- `POST /api/v1/occlusion/analyze`
- Compatibility aliases:
  - `POST /api/occlusion/count`
  - `POST /api/occlusion/analyze`

Example:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/occlusion/analyze" \
  -F "images=@/path/to/image.jpg"
```

## Notes

- Use UTF-8 for `data.yaml` so Chinese class names are preserved in JSON output.
- The visualization renderer uses Pillow and Windows Chinese fonts when
  available, so Chinese labels can be drawn on output images.
- `__pycache__`, virtual environments, runs, outputs, and model weights are
  ignored by Git.
