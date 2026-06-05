"""Depth Anything V2 wrapper for monocular depth estimation.

Usage:
    from occlusion.depth_estimator import DepthEstimator
    estimator = DepthEstimator(encoder="vitb", device="cuda")
    depth_map = estimator.infer(image_bgr)  # numpy array in meters

Weights must be downloaded manually from the official repository:
https://github.com/DepthAnything/Depth-Anything-V2
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

from occlusion.config import (
    DEPTH_ENCODER,
    DEPTH_FEATURES,
    DEPTH_INPUT_SIZE,
    DEPTH_OUT_CHANNELS,
    DEPTH_SCALE_CLIP_MAX,
    DEPTH_SCALE_CLIP_MIN,
    DEPTH_WEIGHTS_DIR,
)


def _build_model(encoder: str, device: str | torch.device):
    """Build DepthAnythingV2 model."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise ImportError(
            "depth_anything_v2 is not installed. "
            "Please install it from https://github.com/DepthAnything/Depth-Anything-V2"
        ) from exc

    model = DepthAnythingV2(
        encoder=encoder,
        features=DEPTH_FEATURES,
        out_channels=DEPTH_OUT_CHANNELS,
    )
    model = model.to(device).eval()
    return model


def _weight_filename(encoder: str) -> str:
    mapping = {
        "vits": "depth_anything_v2_vits.pth",
        "vitb": "depth_anything_v2_vitb.pth",
        "vitl": "depth_anything_v2_vitl.pth",
        "vitg": "depth_anything_v2_vitg.pth",
    }
    return mapping.get(encoder, f"depth_anything_v2_{encoder}.pth")


class DepthEstimator:
    """Monocular depth estimator using Depth Anything V2."""

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = DEPTH_ENCODER,
        device: str | torch.device = "cuda",
        weights_path: str | Path | None = None,
    ) -> None:
        self.encoder = encoder
        self.device = torch.device(device) if isinstance(device, str) else device
        self.input_size = DEPTH_INPUT_SIZE

        self.model = _build_model(encoder, self.device)

        if weights_path is None:
            weights_path = DEPTH_WEIGHTS_DIR / _weight_filename(encoder)
        else:
            weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 weights not found: {weights_path}\n"
                f"Please download them from https://github.com/DepthAnything/Depth-Anything-V2 "
                f"and place under {DEPTH_WEIGHTS_DIR}/ or pass weights_path explicitly."
            )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    @torch.inference_mode()
    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Infer depth map from a BGR image.

        Args:
            image_bgr: uint8 HWC image in BGR order.

        Returns:
            Depth map in meters, float32 HxW.  Values are clipped to a
            reasonable range [0.1, 2.0] meters for retail scenarios.
        """
        h, w = image_bgr.shape[:2]
        # DA-V2 expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        tensor = (
            torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        tensor = tensor.to(self.device)

        depth = self.model(tensor)
        depth = depth.squeeze().cpu().numpy()

        # Resize back to original resolution
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Depth Anything V2 outputs *relative* inverse depth.
        # For metric depth in retail scenes we apply a simple affine scaling
        # (users are encouraged to calibrate with a few measured samples).
        depth = self._convert_to_metric(depth)
        return depth

    def _convert_to_metric(self, raw_depth: np.ndarray) -> np.ndarray:
        """Convert raw relative depth to approximate metric depth (meters).

        Depth Anything V2 raw output is inverse depth up to an unknown scale
        and shift.  For retail shelf scenarios a pragmatic linear mapping is:
            depth_m ≈ 1.0 / (raw_depth * scale + offset)
        Here we use a simple heuristic clipping; users should calibrate
        `self.scale` and `self.offset` with a few ground-truth measurements.
        """
        # Default heuristic: invert and clip to retail-relevant range.
        # This is a placeholder; accurate metric depth requires calibration
        # or using the metric-depth finetuned variants of DA-V2.
        depth = 1.0 / (raw_depth + 1e-6)
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-6)
        depth = depth * (DEPTH_SCALE_CLIP_MAX - DEPTH_SCALE_CLIP_MIN) + DEPTH_SCALE_CLIP_MIN
        return depth.astype(np.float32)

    def calibrate_scale_offset(
        self,
        raw_depths: list[np.ndarray],
        gt_depths: list[np.ndarray],
        masks: list[np.ndarray] | None = None,
    ) -> tuple[float, float]:
        """Calibrate scale and offset using a few ground-truth depth samples.

        Returns (scale, offset) such that:
            metric = 1.0 / (raw_depth * scale + offset)
        """
        raw_vals = []
        gt_vals = []
        for raw, gt in zip(raw_depths, gt_depths):
            m = masks.pop(0) if masks else np.ones_like(raw, dtype=bool)
            raw_vals.extend(raw[m].flatten().tolist())
            gt_vals.extend(gt[m].flatten().tolist())

        raw_vals = np.array(raw_vals)
        gt_vals = np.array(gt_vals)
        inv_gt = 1.0 / (gt_vals + 1e-6)

        # Least squares: inv_gt ≈ raw_vals * scale + offset
        A = np.vstack([raw_vals, np.ones_like(raw_vals)]).T
        scale, offset = np.linalg.lstsq(A, inv_gt, rcond=None)[0]
        return float(scale), float(offset)
