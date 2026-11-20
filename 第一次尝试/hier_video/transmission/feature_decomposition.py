"""Public/residual feature decomposition used by the compression comparison."""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class FeatureMeta:
    original_shape: Tuple[int, int, int, int, int]
    original_dtype: torch.dtype


@dataclass(frozen=True)
class FeatureParts:
    public: torch.Tensor
    individual: torch.Tensor
    meta: FeatureMeta


class PublicResidualDecomposer:
    """Split a BTCHW feature without spatial downsampling.

    The current production path deliberately keeps ``factor=1``: public
    features are the temporal mean and individual features are full-resolution
    residuals. This preserves the spatial information needed by the decoder.
    """

    def __init__(self) -> None:
        pass

    def encode(self, feature: torch.Tensor) -> FeatureParts:
        if not isinstance(feature, torch.Tensor) or feature.ndim != 5:
            raise ValueError("feature must be a rank-5 BTCHW tensor")
        if not feature.is_floating_point():
            raise TypeError(f"Expected floating-point features, got {feature.dtype}")
        public = feature.mean(dim=1, keepdim=True)
        individual = feature - public
        return FeatureParts(
            public=public,
            individual=individual,
            meta=FeatureMeta(
                original_shape=tuple(int(value) for value in feature.shape),
                original_dtype=feature.dtype,
            ),
        )

    def decode(self, parts: FeatureParts) -> torch.Tensor:
        batch, frames, channels, height, width = parts.meta.original_shape
        public = parts.public
        individual = parts.individual
        if tuple(public.shape) != (batch, 1, channels, height, width):
            raise ValueError("public feature shape mismatch")
        if tuple(individual.shape) != (batch, frames, channels, height, width):
            raise ValueError("individual feature shape mismatch")
        if public.device != individual.device:
            raise ValueError("public and individual tensors must share a device")
        restored = public.expand(batch, frames, channels, height, width) + individual
        return restored.to(dtype=parts.meta.original_dtype)


def tensor_payload_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def parts_payload_bytes(parts: FeatureParts) -> int:
    return tensor_payload_bytes(parts.public) + tensor_payload_bytes(parts.individual)
