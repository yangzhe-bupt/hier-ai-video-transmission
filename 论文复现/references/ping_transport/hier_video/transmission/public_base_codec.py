# ======================================================================
# 文件：public_base_codec.py
# 功能：把四帧的公共特征编码成可独立接收的应用层包。
# 函数汇总：
#   - encode_public_gop：量化、打包，并返回四个序列化公共包。
#   - decode_public_gop：仅凭接收包恢复 [1,4,24,32,32] 解码器输入。
# 约定：用于当前 128×128 裁剪与四帧 GOP，个体特征在该档位省略。
# ======================================================================

import math

import torch
import torch.nn.functional as F

from .feature_mapping import build_group_manifest
from .packetization import deserialize_packet, pack_group_tensor, unpack_feature_packets
from .quantization import dequantize_from_metadata, quantize_symmetric


GOP_FRAMES = 4
PUBLIC_SHAPE = (1, 1, 24, 32, 32)
_PROFILES = {
    "int8": ("public-base-int8-v1", PUBLIC_SHAPE, (16, 16), torch.int8),
    "int4": ("public-base-int4-v1", (1, 1, 24, 32, 16), (16, 8), torch.uint8),
    "pool2_int8": ("public-base-pool2-v1", (1, 1, 24, 16, 16), (8, 8), torch.int8),
}


def _manifest(profile):
    """使用已有分组接口规定四个公共区域，个体区域只占映射位置。"""
    version, public_shape, tile, _ = _PROFILES[profile]
    return build_group_manifest(public_shape, (1, GOP_FRAMES, *public_shape[2:]),
                                tile_hw=tile, interface_version=version)


def _pack_int4(public):
    """对称量化到 [-7,7]，将相邻两个值依次写入同一字节的高四位和低四位。"""
    scale = (public.abs().amax() / 7).clamp_min(torch.finfo(public.dtype).eps)
    levels = (torch.round(public / scale).clamp(-7, 7) + 8).to(torch.uint8)
    packed = (levels[..., ::2] << 4) | levels[..., 1::2]
    return packed, {"scheme": "symmetric_packed_nibbles", "bits": 4,
                    "scale": float(scale.item()), "zero_point": 8}


def _unpack_int4(packed, metadata):
    """拆开四位值，按包头中的量化尺度恢复浮点公共特征。"""
    scale = metadata.get("scale")
    if (metadata.get("scheme") != "symmetric_packed_nibbles"
            or metadata.get("bits") != 4 or metadata.get("zero_point") != 8
            or type(scale) not in (float, int) or not math.isfinite(scale) or scale <= 0):
        raise ValueError("int4 公共包量化参数无效")
    high, low = packed >> 4, packed & 15
    if torch.any((high == 0) | (low == 0)).item():
        raise ValueError("int4 公共包包含无效量化码")
    levels = torch.stack((high, low), dim=-1).reshape(PUBLIC_SHAPE)
    return (levels.to(torch.float32) - 8) * scale


@torch.no_grad()
def encode_public_gop(public, profile, *, gop_id, start_frame):
    """
    参数：
        public：时序公共特征张量 [1,1,24,32,32]。
        profile：int8、int4 或 pool2_int8。
        gop_id、start_frame：写入包头的 GOP 标识和源帧起点。
    返回：四个公共包的序列化字节，已包含应用层包头。
    """
    if profile not in _PROFILES:
        raise ValueError("未知公共特征编码档位")
    if tuple(public.shape) != PUBLIC_SHAPE or not public.is_floating_point():
        raise ValueError("公共特征必须为 [1,1,24,32,32] 浮点张量")
    if not torch.isfinite(public).all().item():
        raise ValueError("公共特征包含非有限值")

    if profile == "int4":
        storage, metadata = _pack_int4(public)
    else:
        if profile == "pool2_int8":
            public = F.avg_pool2d(public[:, 0], kernel_size=2).unsqueeze(1)
        quantized = quantize_symmetric(public)
        storage, metadata = quantized.tensor, quantized.metadata

    manifest = _manifest(profile)
    return tuple(
        pack_group_tensor(
            storage, group, tensor_shape=manifest.public_shape,
            stream_id="public-base", sequence_id=gop_id, gop_id=gop_id,
            gop_start_frame=start_frame, packet_id=group.group_id,
            mapping_version=manifest.interface_version, quantization=metadata,
            frame_count=GOP_FRAMES,
        ).serialize()
        for group in manifest.groups if group.branch == "public"
    )


@torch.no_grad()
def decode_public_gop(wire, *, device):
    """
    参数：
        wire：收到的公共包序列化字节。调用方先按包标识去除副本。
        device：解码器所在设备。
    返回：
        (features, status)。四个区域齐全时复制公共特征为四帧输入。
        公共包缺失时返回 (None, 状态)，格式错误或校验失败时抛出异常。
    """
    parsed = tuple(deserialize_packet(raw) for raw in wire)
    if not parsed:
        return None, "missing_all_public"
    first = parsed[0].header
    versions = {config[0]: profile for profile, config in _PROFILES.items()}
    profile = versions.get(first.mapping_version)
    if profile is None:
        raise ValueError("未知公共包版本")
    _, _, _, dtype = _PROFILES[profile]
    expected_dtype = str(dtype).removeprefix("torch.")
    identity = (first.gop_id, first.frame_id_start, first.frame_id_end)
    for packet in parsed:
        header = packet.header
        if (header.stream_id != "public-base" or header.branch != "public"
                or header.sequence_id != header.gop_id or header.group_id != header.packet_id
                or (header.gop_id, header.frame_id_start, header.frame_id_end) != identity
                or header.frame_id_end - header.frame_id_start != GOP_FRAMES
                or header.payload_dtype != expected_dtype or header.quantization is None):
            raise ValueError("公共包的身份、帧数、存储类型或元信息不一致")

    manifest = _manifest(profile)
    received = unpack_feature_packets(parsed, manifest, device=device)
    if any(group.group_id in received.missing_group_ids for group in manifest.groups
           if group.branch == "public"):
        return None, "missing_public_region"

    metadata = received.quantization_by_branch["public"]
    if profile == "int4":
        public = _unpack_int4(received.public, metadata)
    else:
        public = dequantize_from_metadata(received.public, metadata)
        if profile == "pool2_int8":
            public = F.interpolate(public[:, 0], size=PUBLIC_SHAPE[-2:],
                                   mode="bilinear", align_corners=False).unsqueeze(1)
    return public.expand(-1, GOP_FRAMES, -1, -1, -1), "complete_public"

