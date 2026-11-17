"""Versioned feature packet fields and a zero-loss pack/unpack baseline.

The packet header is the contract between compression, the channel simulator,
and the receiver. Payloads are currently raw contiguous tensor bytes. Quantized
payloads can use the same functions by passing an integer tensor and attaching
the quantization parameters in ``quantization``.
"""

from dataclasses import dataclass, field
import hashlib
import json
import struct
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .feature_mapping import FeatureGroup, FeatureGroupManifest, TensorRegion


PACKET_SCHEMA_VERSION = "packet-v1"
NO_INTERLEAVE_VERSION = "none-v1"


_TORCH_TO_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.uint8: "uint8",
}
_NAME_TO_TORCH = {value: key for key, value in _TORCH_TO_NAME.items()}
_NAME_TO_NUMPY = {
    "float32": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "int8": np.dtype("i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "uint8": np.dtype("u1"),
}


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _TORCH_TO_NAME[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported packet dtype: {dtype}") from exc


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _NAME_TO_TORCH[name]
    except KeyError as exc:
        raise TypeError(f"Unsupported packet dtype name: {name!r}") from exc


@dataclass(frozen=True)
class PacketHeader:
    """Fields carried with each independently decodable feature packet."""

    schema_version: str
    stream_id: str
    sequence_id: int
    gop_id: int
    frame_id_start: int
    frame_id_end: int
    packet_id: int
    group_id: int
    branch: str
    layout: str
    tensor_shape: tuple[int, ...]
    tensor_region: TensorRegion
    payload_dtype: str
    payload_shape: tuple[int, ...]
    payload_length: int
    mapping_version: str
    interleave_version: str
    priority: int
    required: bool
    checksum_sha256: str
    quantization: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stream_id": self.stream_id,
            "sequence_id": self.sequence_id,
            "gop_id": self.gop_id,
            "frame_id_start": self.frame_id_start,
            "frame_id_end": self.frame_id_end,
            "packet_id": self.packet_id,
            "group_id": self.group_id,
            "branch": self.branch,
            "layout": self.layout,
            "tensor_shape": list(self.tensor_shape),
            "tensor_region": self.tensor_region.as_dict(),
            "payload_dtype": self.payload_dtype,
            "payload_shape": list(self.payload_shape),
            "payload_length": self.payload_length,
            "mapping_version": self.mapping_version,
            "interleave_version": self.interleave_version,
            "priority": self.priority,
            "required": self.required,
            "checksum_sha256": self.checksum_sha256,
            "quantization": dict(self.quantization) if self.quantization is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PacketHeader":
        return cls(
            schema_version=str(value["schema_version"]),
            stream_id=str(value["stream_id"]),
            sequence_id=int(value["sequence_id"]),
            gop_id=int(value["gop_id"]),
            frame_id_start=int(value["frame_id_start"]),
            frame_id_end=int(value["frame_id_end"]),
            packet_id=int(value["packet_id"]),
            group_id=int(value["group_id"]),
            branch=str(value["branch"]),
            layout=str(value["layout"]),
            tensor_shape=tuple(int(v) for v in value["tensor_shape"]),
            tensor_region=TensorRegion.from_dict(dict(value["tensor_region"])),
            payload_dtype=str(value["payload_dtype"]),
            payload_shape=tuple(int(v) for v in value["payload_shape"]),
            payload_length=int(value["payload_length"]),
            mapping_version=str(value["mapping_version"]),
            interleave_version=str(value["interleave_version"]),
            priority=int(value["priority"]),
            required=bool(value["required"]),
            checksum_sha256=str(value["checksum_sha256"]),
            quantization=value.get("quantization"),
        )


@dataclass(frozen=True)
class FeaturePacket:
    header: PacketHeader
    payload: bytes

    def serialize(self) -> bytes:
        header_bytes = json.dumps(
            self.header.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return struct.pack(">I", len(header_bytes)) + header_bytes + self.payload


@dataclass(frozen=True)
class UnpackedFeatureParts:
    public: torch.Tensor
    individual: torch.Tensor
    received_group_ids: tuple[int, ...]
    missing_group_ids: tuple[int, ...]
    quantization_by_branch: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def deserialize_packet(data: bytes) -> FeaturePacket:
    """Parse a wire packet and verify its declared length and checksum."""
    if len(data) < 4:
        raise ValueError("Packet is shorter than the header-length prefix")
    header_length = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + header_length:
        raise ValueError("Packet ends before the declared header")
    header_end = 4 + header_length
    header = PacketHeader.from_dict(json.loads(data[4:header_end].decode("utf-8")))
    payload = data[header_end:]
    if len(payload) != header.payload_length:
        raise ValueError(
            f"Payload length mismatch: header={header.payload_length}, actual={len(payload)}"
        )
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != header.checksum_sha256:
        raise ValueError(f"Checksum mismatch for packet_id={header.packet_id}")
    return FeaturePacket(header=header, payload=payload)


def _slice_for_region(tensor: torch.Tensor, region: TensorRegion) -> torch.Tensor:
    result = tensor[
        region.b[0] : region.b[1],
        region.t[0] : region.t[1],
        region.c[0] : region.c[1],
        region.h[0] : region.h[1],
        region.w[0] : region.w[1],
    ]
    return result.contiguous()


def pack_group_tensor(
    tensor: torch.Tensor,
    group: FeatureGroup,
    *,
    tensor_shape: tuple[int, ...],
    stream_id: str,
    sequence_id: int,
    gop_id: int,
    gop_start_frame: int = 0,
    packet_id: int,
    mapping_version: str,
    quantization: Mapping[str, Any] | None = None,
    interleave_version: str = NO_INTERLEAVE_VERSION,
    frame_count: int | None = None,
) -> FeaturePacket:
    """Create one packet from the region described by ``group``."""
    if tensor.ndim != 5 or tuple(tensor.shape) != tuple(tensor_shape):
        raise ValueError(f"Tensor shape {tuple(tensor.shape)} does not match {tensor_shape}")
    if group.tensor_region.shape != tuple(_slice_for_region(tensor, group.tensor_region).shape):
        raise ValueError(f"Group {group.group_id} produced an unexpected payload shape")
    payload_tensor = _slice_for_region(tensor, group.tensor_region)
    dtype_name = _dtype_name(payload_tensor.dtype)
    payload = payload_tensor.detach().cpu().numpy().tobytes(order="C")
    region = group.tensor_region
    frame_id_start = gop_start_frame + region.t[0]
    frame_id_end = gop_start_frame + (region.t[1] if frame_count is None else frame_count)
    header = PacketHeader(
        schema_version=PACKET_SCHEMA_VERSION,
        stream_id=stream_id,
        sequence_id=sequence_id,
        gop_id=gop_id,
        frame_id_start=frame_id_start,
        frame_id_end=frame_id_end,
        packet_id=packet_id,
        group_id=group.group_id,
        branch=group.branch,
        layout="BTCHW",
        tensor_shape=tuple(int(v) for v in tensor_shape),
        tensor_region=region,
        payload_dtype=dtype_name,
        payload_shape=tuple(int(v) for v in payload_tensor.shape),
        payload_length=len(payload),
        mapping_version=mapping_version,
        interleave_version=interleave_version,
        priority=group.priority,
        required=group.required,
        checksum_sha256=hashlib.sha256(payload).hexdigest(),
        quantization=quantization,
    )
    return FeaturePacket(header=header, payload=payload)


def build_feature_packets(
    public: torch.Tensor,
    individual: torch.Tensor,
    manifest: FeatureGroupManifest,
    *,
    stream_id: str = "semantic-video",
    sequence_id: int = 0,
    gop_id: int = 0,
    gop_start_frame: int = 0,
    packet_id_start: int = 0,
    quantization_by_branch: Mapping[str, Mapping[str, Any]] | None = None,
    interleave_version: str = NO_INTERLEAVE_VERSION,
) -> list[FeaturePacket]:
    """Build one independently addressable packet per manifest group."""
    manifest.validate()
    if tuple(public.shape) != manifest.public_shape:
        raise ValueError("public tensor shape does not match the mapping manifest")
    if tuple(individual.shape) != manifest.individual_shape:
        raise ValueError("individual tensor shape does not match the mapping manifest")
    quantization_by_branch = quantization_by_branch or {}
    packets: list[FeaturePacket] = []
    for offset, group in enumerate(manifest.groups):
        source = public if group.branch == "public" else individual
        packets.append(
            pack_group_tensor(
                source,
                group,
                tensor_shape=tuple(source.shape),
                stream_id=stream_id,
                sequence_id=sequence_id,
                gop_id=gop_id,
                gop_start_frame=gop_start_frame,
                packet_id=packet_id_start + offset,
                mapping_version=manifest.interface_version,
                quantization=quantization_by_branch.get(group.branch),
                interleave_version=interleave_version,
                frame_count=(int(individual.shape[1]) if group.branch == "public" else None),
            )
        )
    return packets


def _tensor_from_payload(payload: bytes, dtype_name: str, shape: tuple[int, ...]) -> torch.Tensor:
    np_dtype = _NAME_TO_NUMPY.get(dtype_name)
    if np_dtype is None:
        raise TypeError(f"Unsupported payload dtype: {dtype_name!r}")
    expected_bytes = int(np.prod(shape)) * np_dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"Payload bytes {len(payload)} do not match shape {shape} and dtype {dtype_name}")
    array = np.frombuffer(payload, dtype=np_dtype).copy().reshape(shape)
    return torch.from_numpy(array).to(dtype=_dtype_from_name(dtype_name))


def unpack_feature_packets(
    packets: Iterable[FeaturePacket | bytes],
    manifest: FeatureGroupManifest,
    *,
    device: torch.device | str = "cpu",
) -> UnpackedFeatureParts:
    """Reassemble packets and report groups that were not received."""
    manifest.validate()
    group_by_id = {group.group_id: group for group in manifest.groups}
    received: dict[int, FeaturePacket] = {}
    quantization_by_branch: dict[str, Mapping[str, Any]] = {}
    for item in packets:
        packet = deserialize_packet(item) if isinstance(item, bytes) else item
        header = packet.header
        group = group_by_id.get(header.group_id)
        if group is None:
            raise ValueError(f"Packet references unknown group_id={header.group_id}")
        if header.schema_version != PACKET_SCHEMA_VERSION:
            raise ValueError(f"Unsupported packet schema: {header.schema_version!r}")
        if header.mapping_version != manifest.interface_version:
            raise ValueError(
                f"Mapping version mismatch: packet={header.mapping_version!r}, "
                f"manifest={manifest.interface_version!r}"
            )
        if header.branch != group.branch or header.tensor_region != group.tensor_region:
            raise ValueError(f"Packet mapping mismatch for group_id={header.group_id}")
        expected_shape = manifest.public_shape if header.branch == "public" else manifest.individual_shape
        if tuple(header.tensor_shape) != tuple(expected_shape):
            raise ValueError(f"Tensor shape mismatch for group_id={header.group_id}")
        if tuple(header.payload_shape) != tuple(group.tensor_region.shape):
            raise ValueError(f"Payload region shape mismatch for group_id={header.group_id}")
        if header.quantization is not None:
            previous = quantization_by_branch.get(header.branch)
            if previous is not None and dict(previous) != dict(header.quantization):
                raise ValueError(f"Inconsistent quantization metadata for branch={header.branch}")
            quantization_by_branch[header.branch] = dict(header.quantization)
        if header.group_id in received:
            raise ValueError(f"Duplicate packet for group_id={header.group_id}")
        received[header.group_id] = packet

    if not received:
        raise ValueError("At least one packet is required to infer payload dtype")
    first_dtype = next(iter(received.values())).header.payload_dtype
    public = torch.zeros(manifest.public_shape, dtype=_dtype_from_name(first_dtype), device=device)
    individual = torch.zeros(manifest.individual_shape, dtype=_dtype_from_name(first_dtype), device=device)

    for group_id, packet in received.items():
        header = packet.header
        tensor = _tensor_from_payload(packet.payload, header.payload_dtype, header.payload_shape).to(device)
        target = public if header.branch == "public" else individual
        region = header.tensor_region
        target[
            region.b[0] : region.b[1],
            region.t[0] : region.t[1],
            region.c[0] : region.c[1],
            region.h[0] : region.h[1],
            region.w[0] : region.w[1],
        ] = tensor

    received_ids = tuple(sorted(received))
    missing_ids = tuple(group.group_id for group in manifest.groups if group.group_id not in received)
    return UnpackedFeatureParts(
        public=public,
        individual=individual,
        received_group_ids=received_ids,
        missing_group_ids=missing_ids,
        quantization_by_branch=quantization_by_branch,
    )
