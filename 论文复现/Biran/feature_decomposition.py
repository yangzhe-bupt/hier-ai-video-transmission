"""Feature-level public/residual decomposition for BTCHW video tensors.

This is a small, deterministic baseline for interface integration. It is not a
trained codec and does not include quantization, byte serialization, or packetization.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureMeta:
    """Shape information required to restore the original BTCHW layout."""

    original_shape: Tuple[int, int, int, int, int]
    downsample_factor: int
    original_dtype: torch.dtype


@dataclass(frozen=True)
class FeatureParts:
    """Compressed feature representation before quantization/serialization."""

    public: torch.Tensor
    individual: torch.Tensor
    meta: FeatureMeta


class PublicResidualDecomposer:
    """Split a BTCHW feature into a shared temporal component and residuals.

    Baseline definition:
      public[b, 0] = mean(feature[b, t] for t in time)
      individual[b, t] = feature[b, t] - public[b, 0]

    The individual residual is spatially average-pooled by ``downsample_factor``.
    Decoding broadcasts the public component and bilinearly upsamples each
    individual residual before adding them. The output layout matches the input,
    while values are approximate when ``downsample_factor > 1``.
    """

    def __init__(self, downsample_factor: int = 2) -> None:
        if not isinstance(downsample_factor, int) or downsample_factor < 1:
            raise ValueError("downsample_factor must be a positive integer")
        self.downsample_factor = downsample_factor

    def encode(self, feature: torch.Tensor) -> FeatureParts:
        if not isinstance(feature, torch.Tensor):
            raise TypeError("feature must be a torch.Tensor")
        if feature.ndim != 5:
            raise ValueError(
                f"Expected BTCHW rank-5 tensor, got shape {tuple(feature.shape)}"
            )
        if not feature.is_floating_point():
            raise TypeError(f"Expected floating-point features, got {feature.dtype}")

        batch, frames, channels, height, width = feature.shape
        factor = self.downsample_factor
        if height % factor or width % factor:
            raise ValueError(
                f"H and W must be divisible by downsample_factor={factor}; "
                f"got H={height}, W={width}"
            )

        public = feature.mean(dim=1, keepdim=True)
        individual = feature - public

        if factor > 1:
            flat = individual.reshape(batch * frames, channels, height, width)
            flat = F.avg_pool2d(flat, kernel_size=factor, stride=factor)
            individual = flat.reshape(
                batch, frames, channels, height // factor, width // factor
            )

        meta = FeatureMeta(
            original_shape=(batch, frames, channels, height, width),
            downsample_factor=factor,
            original_dtype=feature.dtype,
        )
        return FeatureParts(public=public, individual=individual, meta=meta)

    def decode(self, parts: FeatureParts) -> torch.Tensor:
        if not isinstance(parts, FeatureParts):
            raise TypeError("parts must be a FeatureParts instance")

        batch, frames, channels, height, width = parts.meta.original_shape
        public = parts.public
        individual = parts.individual
        factor = parts.meta.downsample_factor

        if public.ndim != 5 or tuple(public.shape) != (batch, 1, channels, height, width):
            raise ValueError(
                "public feature shape mismatch: expected "
                f"{(batch, 1, channels, height, width)}, got {tuple(public.shape)}"
            )
        if individual.ndim != 5:
            raise ValueError("individual feature must be a rank-5 BTCHW tensor")
        if tuple(individual.shape[:3]) != (batch, frames, channels):
            raise ValueError(
                "individual B/T/C dimensions do not match the original feature shape"
            )
        if public.device != individual.device:
            raise ValueError("public and individual tensors must be on the same device")

        if factor > 1:
            low_h, low_w = individual.shape[-2:]
            flat = individual.reshape(batch * frames, channels, low_h, low_w)
            flat = F.interpolate(
                flat,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            individual = flat.reshape(batch, frames, channels, height, width)

        restored = public.expand(batch, frames, channels, height, width) + individual
        return restored.to(dtype=parts.meta.original_dtype)


def tensor_payload_bytes(tensor: torch.Tensor) -> int:
    """Return raw tensor payload size; excludes container and protocol overhead."""
    return tensor.numel() * tensor.element_size()


def parts_payload_bytes(parts: FeatureParts) -> int:
    """Return raw public+individual tensor bytes; excludes metadata and headers."""
    return tensor_payload_bytes(parts.public) + tensor_payload_bytes(parts.individual)
