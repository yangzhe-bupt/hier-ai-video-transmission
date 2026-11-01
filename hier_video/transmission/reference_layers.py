"""Trainable reference mapping transcribed from the supplied compression checklist."""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class Backend(Protocol):
    backend_id: str

    def split(self, z: torch.Tensor, factor: int) -> "Layers": ...
    def merge(self, parts: "Layers", mode: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class Layers:
    base: torch.Tensor
    enhancement: torch.Tensor
    shape: tuple[int, ...]
    factor: int
    backend: Backend


def validate_feature(z: torch.Tensor, factor: int) -> None:
    if (
        not isinstance(z, torch.Tensor) or z.ndim != 5
        or z.dtype != torch.float32 or any(n <= 0 for n in z.shape)
        or not torch.isfinite(z).all().item()
        or type(factor) is not int or factor < 1
        or z.shape[-2] % factor or z.shape[-1] % factor
    ):
        raise ValueError("finite float32 BTCHW with spatial dimensions divisible by factor required")


class ReferenceBackend:
    backend_id = "reference-public-pooled-v0"

    def split(self, z: torch.Tensor, factor: int) -> Layers:
        validate_feature(z, factor)
        b, t, c, h, w = z.shape
        public = z.mean(dim=1, keepdim=True)
        residual = (z - public).reshape(b * t, c, h, w)
        pooled = F.avg_pool2d(residual, factor, factor)
        return Layers(public, pooled.reshape(b, t, c, h // factor, w // factor),
                      tuple(z.shape), factor, self)

    def merge(self, parts: Layers, mode: str) -> torch.Tensor:
        if mode not in ("base", "full"):
            raise ValueError("allowed actions are base and full, never enhancement only")
        base = parts.base.expand(parts.shape)
        if mode == "base":
            return base
        b, t, c, h, w = parts.shape
        residual = F.interpolate(
            parts.enhancement.reshape(b * t, c, h // parts.factor, w // parts.factor),
            size=(h, w), mode="bilinear", align_corners=False,
        ).reshape(parts.shape)
        return base + residual


def split_layers(z: torch.Tensor, factor: int = 2,
                 backend: Backend | None = None) -> Layers:
    validate_feature(z, factor)
    return (backend or ReferenceBackend()).split(z, factor)


def restore_layers(parts: Layers, mode: str) -> torch.Tensor:
    return parts.backend.merge(parts, mode)
