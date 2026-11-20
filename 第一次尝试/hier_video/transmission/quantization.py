"""Low precision feature quantization for the packetized path."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    """A tensor payload plus the parameters needed for dequantization."""

    tensor: torch.Tensor
    scale: torch.Tensor
    bits: int
    zero_point: int = 0

    @property
    def metadata(self) -> dict[str, Any]:
        if self.scale.numel() != 1:
            raise ValueError("The current packet baseline uses one scale per tensor")
        return {
            "scheme": "symmetric",
            "bits": self.bits,
            "scale": float(self.scale.detach().cpu().item()),
            "zero_point": self.zero_point,
        }


def quantize_symmetric(tensor: torch.Tensor, *, bits: int = 8) -> QuantizedTensor:
    """Quantize a floating tensor to signed int8 storage.

    ``bits=8`` is the current byte-aligned baseline. The scale is stored in the
    packet metadata and the reconstructed tensor remains floating point before
    it enters the decoder.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("quantize_symmetric expects a floating-point tensor")
    if bits != 8:
        raise ValueError("Only bits=8 is implemented in the byte-aligned baseline")
    qmax = (1 << (bits - 1)) - 1
    max_abs = tensor.detach().abs().amax()
    scale = max_abs / qmax
    scale = scale.clamp_min(torch.finfo(tensor.dtype).eps)
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax).to(torch.int8)
    return QuantizedTensor(tensor=quantized, scale=scale.detach(), bits=bits)


def dequantize_symmetric(quantized: QuantizedTensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Restore a quantized tensor to the floating dtype used by the decoder."""
    if quantized.tensor.dtype != torch.int8:
        raise TypeError(f"Expected int8 payload, got {quantized.tensor.dtype}")
    scale = quantized.scale.to(device=quantized.tensor.device, dtype=dtype)
    return (quantized.tensor.to(dtype=dtype) - quantized.zero_point) * scale


def dequantize_from_metadata(
    tensor: torch.Tensor,
    metadata: dict[str, Any],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a received payload using the packet-header metadata."""
    if metadata.get("scheme") != "symmetric" or int(metadata.get("bits", 0)) != 8:
        raise ValueError(f"Unsupported quantization metadata: {metadata}")
    if tensor.dtype != torch.int8:
        raise TypeError(f"Expected int8 received payload, got {tensor.dtype}")
    scale = torch.as_tensor(metadata["scale"], device=tensor.device, dtype=dtype)
    if scale.numel() != 1 or not torch.isfinite(scale).item() or scale.item() <= 0:
        raise ValueError("Quantization scale must be one positive finite value")
    zero_point = int(metadata.get("zero_point", 0))
    return (tensor.to(dtype=dtype) - zero_point) * scale


def quantized_payload_bytes(quantized: QuantizedTensor) -> int:
    return quantized.tensor.numel() * quantized.tensor.element_size()
