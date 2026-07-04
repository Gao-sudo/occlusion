"""Default configuration for the occlusion counting pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project paths (resolved relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_occlusion"
DEFAULT_DATA_YAML = PROJECT_ROOT / "data_occlusion" / "data.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "occlusion"
DEFAULT_PRETRAINED_SEG = PROJECT_ROOT / "yolo11s-seg.pt"  # official Ultralytics seg weights

# ---------------------------------------------------------------------------
# Depth Anything V2 settings
# ---------------------------------------------------------------------------
DEPTH_ENCODER: str = "vitb"           # options: vits, vitb, vitl, vitg
DEPTH_FEATURES: int = 128
DEPTH_OUT_CHANNELS: list[int] = [64, 128, 256, 512]
DEPTH_INPUT_SIZE: int = 518           # DA-V2 default inference size
DEPTH_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "depth_anything_v2"

# ---------------------------------------------------------------------------
# Inference / training defaults
# ---------------------------------------------------------------------------
DEFAULT_DEVICE: str = "0"
DEFAULT_IMGSZ: int = 640
DEFAULT_CONF: float = 0.25
DEFAULT_IOU: float = 0.5              # slightly lower than detect default to preserve overlaps
DEFAULT_MAX_DET: int = 300
DEFAULT_LINE_WIDTH: int = 2
DEFAULT_FONT_SCALE: float = 0.5
DEFAULT_BATCH: int = 8

# ---------------------------------------------------------------------------
# Fusion counter parameters
# ---------------------------------------------------------------------------
# SKU thickness priors (meters).  These are placeholders and should be
# calibrated per SKU with real measurements or online statistics.
DEFAULT_SKU_SPEC: dict[str, dict[str, Any]] = {
    "九牧增压花洒": {"unit_depth_m": 0.045, "unit_mask_area_px": None},
    "九牧增压花洒套装": {"unit_depth_m": 0.060, "unit_mask_area_px": None},
    "九牧大冲力喷枪角阀": {"unit_depth_m": 0.035, "unit_mask_area_px": None},
    "九牧安全快开": {"unit_depth_m": 0.030, "unit_mask_area_px": None},
    "九牧安全角阀": {"unit_depth_m": 0.032, "unit_mask_area_px": None},
    "九牧百搭下水": {"unit_depth_m": 0.040, "unit_mask_area_px": None},
    "九牧百搭下水（软袋）": {"unit_depth_m": 0.025, "unit_mask_area_px": None},
    "九牧轻音盖板": {"unit_depth_m": 0.050, "unit_mask_area_px": None},
    "九牧防断裂淋浴软管": {"unit_depth_m": 0.035, "unit_mask_area_px": None},
    "九牧防漏水件": {"unit_depth_m": 0.020, "unit_mask_area_px": None},
    "九牧防爆编织软管": {"unit_depth_m": 0.030, "unit_mask_area_px": None},
    "九牧防臭下水管": {"unit_depth_m": 0.040, "unit_mask_area_px": None},
    "九牧防臭地漏": {"unit_depth_m": 0.025, "unit_mask_area_px": None},
    "九牧健康编织软管": {"unit_depth_m": 0.030, "unit_mask_area_px": None},
}

# Clustering / projection
CLUSTER_EPS_PX: float = 40.0          # DBSCAN eps for grouping masks on the same hook/cluster
CLUSTER_MIN_SAMPLES: int = 1
DEPTH_STEP_THRESHOLD: float = 0.015   # meters; depth discontinuity considered a step
DEPTH_SCALE_CLIP_MIN: float = 0.1     # min depth in meters (clipping)
DEPTH_SCALE_CLIP_MAX: float = 2.0     # max depth in meters (clipping)
COUNT_DIFF_TOLERANCE: int = 1         # allowed diff between visible_count and estimated_total for "high" confidence

# ---------------------------------------------------------------------------
# Countable / uncountable classification
# ---------------------------------------------------------------------------
UNCOUNTABLE_MASK_IOU_THRESHOLD: float = 0.30      # max allowed IoU inside a countable cluster
UNCOUNTABLE_STEP_RATIO_THRESHOLD: float = 1.5     # steps / visible_count > this -> uncountable
UNCOUNTABLE_DENSITY_MASKS_PER_M: float = 50.0     # masks per meter of depth > this -> uncountable
UNCOUNTABLE_MIN_CONFIDENCE_RATIO: float = 0.50    # fraction of masks with conf < 0.5 -> uncountable
UNCOUNTABLE_MIN_VISIBLE_FOR_REFERENCE: int = 2    # min visible items to learn unit depth

# ---------------------------------------------------------------------------
# Context-aware instance decision thresholds (no-depth mode)
# ---------------------------------------------------------------------------
DECISION_CONFIRMED_THRESHOLD: float = 0.80        # confidence >= this -> confirmed
DECISION_CONTEXT_MIN_CONF: float = 0.25           # min confidence to be considered for context support
DECISION_CONTEXT_MAX_CONF: float = 0.80           # max confidence for context-support tier
DECISION_VERTICAL_ASPECT_MIN: float = 1.5         # mask height / width >= this -> vertical shape
DECISION_CLUSTER_AXIS_ALIGN_DEG: float = 30.0     # mask centroid within this angle of cluster axis
