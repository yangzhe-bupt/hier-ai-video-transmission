# ======================================================================
# 文件：layered_public_codec.py
# 功能：为四帧公共基础档加入一个低码率个体残差包。
# 函数汇总：
#   - encode_individual_enhancement：池化并量化前三帧个体残差。
#   - decode_layered_gop：先恢复公共特征，增强包可用时再融合残差。
# 约定：第 4 帧残差由前三帧残差的负和恢复。增强包独立记账。
# ======================================================================

import math

import torch
import torch.nn.functional as F

from .feature_mapping import FeatureGroup, TensorRegion
from .packetization import PACKET_SCHEMA_VERSION, deserialize_packet, pack_group_tensor
from .public_base_codec import GOP_FRAMES, PUBLIC_SHAPE, decode_public_gop
from .quantization import dequantize_from_metadata, quantize_symmetric


_VERSION = "public-residual-pool8-v1"
_GROUP = FeatureGroup(
    4, "individual", TensorRegion((0, 1), (0, GOP_FRAMES - 1), (0, 24), (0, 4), (0, 4)),
    priority=1, required=False,
)
_RESIDUAL_SHAPE = (1, GOP_FRAMES, *PUBLIC_SHAPE[2:])


@torch.no_grad()
def encode_individual_enhancement(residual, *, gop_id, start_frame):
    """
    参数：
        residual：公共均值分解产生的 [1,4,24,32,32] 个体特征。
        gop_id、start_frame：与该 GOP 公共包一致的身份字段。
    返回：一个序列化应用层包，包含前三帧 4×4 int8 残差和量化参数。
    """
    if tuple(residual.shape) != _RESIDUAL_SHAPE or not residual.is_floating_point():
        raise ValueError("个体残差必须是 [1,4,24,32,32] 浮点张量")
    sampled = residual[0, :GOP_FRAMES - 1]
    if not torch.isfinite(sampled).all().item():
        raise ValueError("个体残差包含非有限值")
    pooled = F.avg_pool2d(sampled, kernel_size=8).unsqueeze(0)
    quantized = quantize_symmetric(pooled)
    return pack_group_tensor(
        quantized.tensor, _GROUP, tensor_shape=_GROUP.tensor_region.shape,
        stream_id="public-base", sequence_id=gop_id, gop_id=gop_id, gop_start_frame=start_frame,
        packet_id=_GROUP.group_id, mapping_version=_VERSION, quantization=quantized.metadata,
        frame_count=GOP_FRAMES,
    ).serialize()


def _decode_enhancement(wire, public_header, device):
    """校验增强包大小、量化元信息及 GOP 归属，再恢复四帧残差。"""
    packet = deserialize_packet(wire)
    header = packet.header
    shape = _GROUP.tensor_region.shape
    if (header.schema_version != PACKET_SCHEMA_VERSION or header.mapping_version != _VERSION
            or header.stream_id != public_header.stream_id or header.sequence_id != public_header.sequence_id
            or header.gop_id != public_header.gop_id or header.frame_id_start != public_header.frame_id_start
            or header.frame_id_end != public_header.frame_id_end or header.branch != _GROUP.branch
            or header.packet_id != _GROUP.group_id or header.group_id != _GROUP.group_id
            or header.tensor_region != _GROUP.tensor_region or header.tensor_shape != shape
            or header.payload_shape != shape or header.payload_dtype != "int8"
            or len(packet.payload) != math.prod(shape)):
        raise ValueError("个体残差包与当前 GOP 或格式不匹配")
    metadata = header.quantization
    if not isinstance(metadata, dict) or metadata.get("zero_point") != 0:
        raise ValueError("个体残差包量化参数无效")
    quantized = torch.frombuffer(bytearray(packet.payload), dtype=torch.int8).reshape(shape).to(device)
    pooled = dequantize_from_metadata(quantized, metadata)
    # 池化和上采样为线性操作。用负和补齐第四帧，保持残差时间均值为零。
    pooled = torch.cat((pooled, -pooled.sum(dim=1, keepdim=True)), dim=1)
    return F.interpolate(pooled[0], size=PUBLIC_SHAPE[-2:], mode="bilinear", align_corners=False).unsqueeze(0)


@torch.no_grad()
def decode_layered_gop(public_wire, enhancement_wire=None, *, device):
    """
    参数：
        public_wire：同一 GOP 已收到并去重的公共包。
        enhancement_wire：已收到的增强包，缺失时传 None。
    返回：
        (features, status)。公共特征可独立使用，增强缺失或格式无效时返回公共特征。
        公共区域未收齐时返回 None，增强包无法单独解码。
    """
    public_packets = tuple(public_wire)
    base, status = decode_public_gop(public_packets, device=device)
    if base is None:
        return None, status
    if enhancement_wire is None:
        return base, "base_only"
    public_header = deserialize_packet(public_packets[0]).header
    try:
        enhancement = _decode_enhancement(enhancement_wire, public_header, device)
    except (ValueError, TypeError, KeyError, OverflowError):
        return base, "base_only_invalid_enhancement"
    return base + enhancement, "with_enhancement"
