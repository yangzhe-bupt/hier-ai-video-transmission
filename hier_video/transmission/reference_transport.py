"""Experimental float32 datagrams with in-band config and receiver-only masks.

The format is local, unauthenticated, and not RTP compatible. No quantizer or
entropy coder is claimed. Config, fragment headers and configured UDP/IP costs
are counted by the network binding.
"""

from dataclasses import dataclass
import hashlib
import json
from math import prod
import struct
import zlib

import numpy as np
import torch

from .reference_layers import Layers, ReferenceBackend, restore_layers


HEADER = struct.Struct("!4sBBIIIII")
BRANCHES = ("config", "base", "enhancement")
MAX_BRANCH_BYTES = 64 * 1024 * 1024


def _uint(value: int) -> bool:
    return type(value) is int and 0 <= value < 2**32


def _ids(session: int, gop: int, branch: str, offset: int) -> tuple[str, str]:
    group_id = f"{session}:{gop}:{branch}"
    return f"{group_id}:{offset}", group_id


@dataclass(frozen=True)
class Datagram:
    packet_id: str
    group_id: str
    branch: str
    data: bytes


@dataclass(frozen=True)
class Offer:
    packets: tuple[Datagram, ...]
    config: dict


@dataclass(frozen=True)
class Reception:
    features: torch.Tensor
    base: torch.Tensor
    applied_enhancement: torch.Tensor
    base_missing: torch.Tensor
    enhancement_missing: torch.Tensor
    state: str
    config: dict

    @property
    def missing_base_elements(self) -> int:
        return int(self.base_missing.sum().item())

    @property
    def missing_enhancement_elements(self) -> int:
        return int(self.enhancement_missing.sum().item())


def _float_bytes(z: torch.Tensor) -> bytes:
    return z.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()


def make_offer(parts: Layers, session_id: int = 1, gop_id: int = 0,
               model_id: str = "untrained-cnn-seed24", mtu: int = 1200) -> Offer:
    if (
        not _uint(session_id) or not _uint(gop_id)
        or not isinstance(model_id, str) or not model_id.strip()
        or type(mtu) is not int or not HEADER.size + 4 <= mtu <= 65507
        or parts.backend.backend_id != ReferenceBackend.backend_id
    ):
        raise ValueError("invalid stream/model/MTU or unsupported backend format")
    payloads = {"base": _float_bytes(parts.base),
                "enhancement": _float_bytes(parts.enhancement)}
    if any(len(v) > MAX_BRANCH_BYTES for v in payloads.values()):
        raise ValueError("reference branch exceeds memory bound")
    config = {
        "schema": "ping-float-v1", "layout": "BTCHW", "dtype": "float32-le",
        "session_id": session_id, "gop_id": gop_id, "model_id": model_id,
        "backend": parts.backend.backend_id, "input_shape": list(parts.shape),
        "factor": parts.factor, "base_shape": list(parts.base.shape),
        "enhancement_shape": list(parts.enhancement.shape),
        "bytes": {key: len(value) for key, value in payloads.items()},
        "sha256": {key: hashlib.sha256(value).hexdigest()
                   for key, value in payloads.items()},
    }
    payloads["config"] = json.dumps(config, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
    chunk_size = (mtu - HEADER.size) // 4 * 4
    packets = []
    for kind, branch in enumerate(BRANCHES):
        raw = payloads[branch]
        for offset in range(0, len(raw), chunk_size):
            chunk = raw[offset:offset + chunk_size]
            prefix = HEADER.pack(b"PING", 1, kind, session_id, gop_id, offset, len(raw), 0)
            crc = zlib.crc32(prefix[:-4] + chunk)
            header = prefix[:-4] + struct.pack("!I", crc)
            packet_id, group_id = _ids(session_id, gop_id, branch, offset)
            packets.append(Datagram(packet_id, group_id, branch, header + chunk))
    return Offer(tuple(packets), config)


def unpack_packet(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or len(raw) <= HEADER.size:
        raise ValueError("truncated datagram")
    magic, version, kind, session, gop, offset, total, crc = HEADER.unpack_from(raw)
    payload = raw[HEADER.size:]
    if (
        magic != b"PING" or version != 1 or kind >= len(BRANCHES)
        or not 0 < total <= MAX_BRANCH_BYTES
        or offset + len(payload) > total
        or zlib.crc32(raw[:HEADER.size - 4] + payload) != crc
        or (kind != 0 and (offset % 4 or len(payload) % 4 or total % 4))
    ):
        raise ValueError("invalid datagram identity, bounds, float alignment or CRC")
    return dict(branch=BRANCHES[kind], session=session, gop=gop,
                offset=offset, total=total, payload=payload)


def selected_packets(offer: Offer, mode: str) -> tuple[Datagram, ...]:
    if mode not in ("base", "full"):
        raise ValueError("only base and full are valid actions")
    return tuple(p for p in offer.packets if mode == "full" or p.branch != "enhancement")


def _join(rows: list[dict], total: int) -> tuple[bytes, np.ndarray]:
    data = bytearray(total)
    present = np.zeros(total, dtype=bool)
    for row in rows:
        start, end = row["offset"], row["offset"] + len(row["payload"])
        if row["total"] != total or end > total or present[start:end].any():
            raise ValueError("overlap or inconsistent fragment length")
        data[start:end] = row["payload"]
        present[start:end] = True
    return bytes(data), present


def _validate_config(config: dict, session: int, gop: int) -> None:
    shape = config.get("input_shape")
    factor = config.get("factor")
    if (
        config.get("schema") != "ping-float-v1" or config.get("layout") != "BTCHW"
        or config.get("dtype") != "float32-le"
        or config.get("backend") != ReferenceBackend.backend_id
        or config.get("session_id") != session or config.get("gop_id") != gop
        or not isinstance(config.get("model_id"), str) or not config["model_id"].strip()
        or not isinstance(shape, list) or len(shape) != 5
        or any(type(n) is not int or n <= 0 for n in shape)
        or prod(shape) * 4 > MAX_BRANCH_BYTES
        or type(factor) is not int or factor < 1
        or shape[-2] % factor or shape[-1] % factor
    ):
        raise ValueError("invalid in-band tensor config")
    b, t, c, h, w = shape
    expected = {"base": [b, 1, c, h, w],
                "enhancement": [b, t, c, h // factor, w // factor]}
    for branch, dims in expected.items():
        length = prod(dims) * 4
        digest = config.get("sha256", {}).get(branch)
        if (
            config.get(f"{branch}_shape") != dims
            or config.get("bytes", {}).get(branch) != length
            or length > MAX_BRANCH_BYTES
            or not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError("config shape, size or hash mismatch")


def receive(packets: list[Datagram] | tuple[Datagram, ...], *,
            expected_session: int | None = None, expected_gop: int | None = None,
            expected_model: str | None = None) -> Reception:
    """Use only received bytes; no offer, source tensor or missing payload argument."""
    rows = {branch: [] for branch in BRANCHES}
    seen = {}
    stream = None
    for packet in packets:
        row = unpack_packet(packet.data)
        key, group = _ids(row["session"], row["gop"], row["branch"], row["offset"])
        if packet.packet_id != key or packet.group_id != group or packet.branch != row["branch"]:
            raise ValueError("external packet labels do not match bytes")
        current = (row["session"], row["gop"])
        if stream is not None and current != stream:
            raise ValueError("mixed stream or GOP")
        stream = current
        if key in seen:
            if seen[key] != packet.data:
                raise ValueError("conflicting duplicate")
            continue
        seen[key] = packet.data
        rows[row["branch"]].append(row)
    if not rows["config"]:
        raise ValueError("in-band config unavailable; GOP cannot be decoded")
    raw, present = _join(rows["config"], rows["config"][0]["total"])
    if not present.all():
        raise ValueError("in-band config incomplete; GOP cannot be decoded")
    try:
        config = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid config JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    _validate_config(config, *stream)
    for actual, expected in ((stream[0], expected_session), (stream[1], expected_gop),
                             (config["model_id"], expected_model)):
        if expected is not None and actual != expected:
            raise ValueError("receiver stream/model binding mismatch")
    tensors, masks = {}, {}
    for branch in ("base", "enhancement"):
        data, valid = _join(rows[branch], config["bytes"][branch])
        if valid.all() and hashlib.sha256(data).hexdigest() != config["sha256"][branch]:
            raise ValueError("complete branch hash mismatch")
        dims = config[f"{branch}_shape"]
        values = np.frombuffer(data, dtype="<f4").copy().reshape(dims)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite float payload")
        tensors[branch] = torch.from_numpy(values)
        masks[branch] = torch.from_numpy((~valid.reshape(-1, 4).all(1)).reshape(dims).copy())
    base_ok = not masks["base"].any().item()
    enhancement = tensors["enhancement"] if base_ok else torch.zeros_like(tensors["enhancement"])
    parts = Layers(tensors["base"], enhancement, tuple(config["input_shape"]),
                   config["factor"], ReferenceBackend())
    state = (
        "base_incomplete" if not base_ok else
        "base" if masks["enhancement"].all().item() else
        "partial_full" if masks["enhancement"].any().item() else "full"
    )
    return Reception(restore_layers(parts, "full"), tensors["base"], enhancement,
                     masks["base"], masks["enhancement"], state, config)
