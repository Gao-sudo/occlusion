"""Depth Anything V2 wrapper for monocular depth estimation.

Usage:
    from occlusion.depth_estimator import DepthEstimator
    estimator = DepthEstimator(encoder="vitb", device="cuda")
    depth_map = estimator.infer(image_bgr)  # numpy array in meters

Weights can be downloaded manually from the official repository:
https://github.com/DepthAnything/Depth-Anything-V2

If the official ``depth_anything_v2`` package is not installed, this module
falls back to the Hugging Face ``transformers`` depth-estimation pipeline
using the ``depth-anything/Depth-Anything-V2-*-hf`` models.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image

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


def _hf_model_name(encoder: str) -> str:
    """Map local encoder name to a HuggingFace depth-estimation model.

    Prefer the metric-indoor variants because retail shelf images are indoor
    scenes and the metric models return absolute depth in meters.
    """
    mapping = {
        "vits": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "vitg": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    }
    return mapping.get(encoder, mapping["vitb"])


class DepthEstimator:
    """Monocular depth estimator using Depth Anything V2.

    Tries to load the official ``depth_anything_v2.dpt.DepthAnythingV2`` model
    first.  If the package or the local checkpoint is missing, it falls back
    to the Hugging Face ``transformers`` depth-estimation pipeline so that
    users can run depth-aware counting without manually cloning the official
    repository.
    """

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"] = DEPTH_ENCODER,
        device: str | torch.device = "cuda",
        weights_path: str | Path | None = None,
    ) -> None:
        self.encoder = encoder
        if isinstance(device, str):
            device = device.strip()
            if device.isdigit():
                device = f"cuda:{device}"
            elif device.lower() in ("cuda", "gpu"):
                device = "cuda:0"
            elif device.lower() == "cpu":
                device = "cpu"
        self.device = torch.device(device)
        self.input_size = DEPTH_INPUT_SIZE

        # Try the official implementation first.
        self._official = False
        self._hf_pipe = None
        self.model = None

        if weights_path is None:
            weights_path = DEPTH_WEIGHTS_DIR / _weight_filename(encoder)
        else:
            weights_path = Path(weights_path)

        try:
            self.model = _build_model(encoder, self.device)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Depth Anything V2 weights not found: {weights_path}\n"
                    f"Please download them from https://github.com/DepthAnything/Depth-Anything-V2 "
                    f"and place under {DEPTH_WEIGHTS_DIR}/ or pass weights_path explicitly."
                )
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self._official = True
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            # Fall back to Hugging Face transformers pipeline.
            warnings.warn(
                f"Official Depth-Anything-V2 implementation unavailable ({exc}). "
                f"Falling back to Hugging Face transformers pipeline "
                f"({_hf_model_name(encoder)})."
            )
            try:
                from transformers import pipeline
            except ImportError as imp_exc:
                raise ImportError(
                    "Neither depth_anything_v2 nor transformers is installed. "
                    "Install one of them to use depth estimation."
                ) from imp_exc

            self._hf_model_name = _hf_model_name(encoder)
            self._hf_pipe = pipeline(
                task="depth-estimation",
                model=self._hf_model_name,
                device=self.device,
            )
            self._hf_is_metric = "metric" in self._hf_model_name.lower()
            self.model = None

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

        if self._official:
            # Official model path.
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            tensor = (
                torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            )
            tensor = tensor.to(self.device)
            depth = self.model(tensor)
            depth = depth.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # HuggingFace pipeline path.
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            result = self._hf_pipe(image_pil)
            # result contains "predicted_depth" tensor and "depth" PIL image.
            predicted = result.get("predicted_depth")
            if predicted is None:
                # Fallback: convert the returned PIL depth image.
                depth = np.asarray(result["depth"], dtype=np.float32)
            else:
                if isinstance(predicted, torch.Tensor):
                    predicted = predicted.squeeze().cpu().numpy()
                depth = np.asarray(predicted, dtype=np.float32)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        if self._hf_pipe is not None and self._hf_is_metric:
            # HuggingFace Metric-Indoor models already return absolute depth in meters.
            depth = np.clip(depth, DEPTH_SCALE_CLIP_MIN, DEPTH_SCALE_CLIP_MAX)
            return depth.astype(np.float32)
        return self._convert_to_metric(depth)

    def _convert_to_metric(self, raw_depth: np.ndarray) -> np.ndarray:
        """Convert raw relative depth to approximate metric depth (meters).

        Depth Anything V2 raw output is inverse depth up to an unknown scale
        and shift.  For retail shelf scenarios a pragmatic linear mapping is:
            depth_m ≈ 1.0 / (raw_depth * scale + offset)
        Here we use a simple heuristic clipping; users should calibrate
        `self.scale` and `self.offset` with a few ground-truth measurements.
        """
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
