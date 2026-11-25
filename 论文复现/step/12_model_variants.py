"""仅改变图7公共残差支路的受控版本；不宣称完整论文复现。"""
import importlib
import torch
from torch import nn

base = importlib.import_module("12_train_stage1")


class Figure7CommonExtractor(nn.Module):
    """保留原时间压缩和通道混合，残差支路改为2D卷积及两处GELU。

    输入BTCHW → 时间压缩为1 → 去掉单例T维 → 2D残差 → 恢复BTCHW。
    原Conv3d在T=1、padding=1时仅中间时间切片有效。复制随机初始化的
    中心切片，使两版本初始有效卷积权重相同，减少无关初始化差异。
    这不是加载已训练权重；新实验仍从随机初始化开始。
    """
    def __init__(self, original):
        super().__init__()
        self.channels = original.channels
        self.gop_size = original.gop_size
        self.temporal_compression = original.temporal_compression
        self.activation = original.activation
        self.channel_mixing = original.channel_mixing
        # 构造新层时恢复全局RNG，避免改变后续量化噪声序列。
        with torch.random.fork_rng(devices=[]):
            self.refinement = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.channels, self.channels, 3, padding=1),
                nn.GELU(),
            )
        with torch.no_grad():
            for index in (0, 2):
                source = original.refinement[index]
                self.refinement[index].weight.copy_(source.weight[:, :, 1])
                self.refinement[index].bias.copy_(source.bias)

    def forward(self, features):
        if features.ndim != 5 or features.shape[1:3] != (self.gop_size, self.channels):
            raise ValueError("需要[B,gop_size,channels,H,W]")
        x = features.permute(0, 2, 1, 3, 4).contiguous()
        x = self.channel_mixing(self.activation(self.temporal_compression(x)))
        if x.shape[2] != 1:
            raise ValueError("公共分支时间维必须压缩为1")
        spatial = x.squeeze(2)
        refined = spatial + self.refinement(spatial)
        return refined.unsqueeze(1)  # [B,1,C,H,W]


class Stage6BlockPair(nn.Module):
    """Stage6两个独立块；单空间窗口时第二块使用非移位注意力。

    当前模型要求正方形空间窗口且输入尺寸为256的倍数。若只有一个轴
    大于窗口，现有掩码不支持单轴移位，明确报错而不悄悄改变规则。
    两块辅助损失取均值，外层再按真实块数加权。
    """
    def __init__(self, first):
        super().__init__()
        self.first = first
        self.second = base.step07.SwinBlock3D(
            dim=144, num_heads=3, window_size=(4,4,4), shift_size=(0,0,0))
        base.step11.replace_mlp_with_moe(self.second, num_experts=4, top_k=2)

    def forward(self, features, return_aux=False):
        h, w = features.shape[-2:]
        if (h > 4) != (w > 4):
            raise ValueError("Stage6当前不支持单轴多窗口，需扩展掩码后使用")
        # second的shift_size只含运行时几何配置，不参与state_dict。
        # 单窗口的4×4输入不移位；8×8及以上启用空间移位。
        self.second.shift_size = (0,2,2) if h > 4 and w > 4 else (0,0,0)
        if return_aux:
            x, a = self.first(features, return_aux=True)
            x, b = self.second(x, return_aux=True)
            return x, (a+b)/2
        return self.second(self.first(features))


class SpatialOnlyDiagnostic(nn.Module):
    """逐帧空间编解码，返回完整模型兼容的三项接口。

    零辅助项/裁剪值仅为日志接口占位，不代表实际存在MoE或量化。
    state_dict布局与Step14空间诊断一致。
    """
    def __init__(self):
        super().__init__()
        self.encoder=base.step06.SpatialEncoder()
        self.decoder=base.step06.SpatialDecoder()

    def forward(self,video):
        b,t,c,h,w=video.shape
        result=self.decoder(self.encoder(video.reshape(b*t,c,h,w))).reshape(b,t,c,h,w)
        return result,result.new_zeros(()),{'common':0.,'individual':0.}


class SpatiotemporalOnlyDiagnostic(nn.Module):
    """空间→时空编码→时空解码→空间解码，无分解、量化或通道。

    先构造两个空间模块，使相同种子下其初始权重与spatial_only一致。
    时空前馈仍使用MoE，辅助损失按8块平均。此诊断同时增加降采样、
    注意力和MoE，不能只凭差值断定某一个算子有问题。
    """
    def __init__(self):
        super().__init__()
        self.spatial_encoder=base.step06.SpatialEncoder()
        self.spatial_decoder=base.step06.SpatialDecoder()
        self.temporal_encoder=base.step07.SpatiotemporalEncoder()
        self.temporal_decoder=base.step07.SpatiotemporalDecoder()
        counts=[base.step11.replace_mlp_with_moe(m,num_experts=4,top_k=2)
                for m in (self.temporal_encoder,self.temporal_decoder)]
        if counts!=[4,4]:raise ValueError('时空MoE块数错误')

    def forward(self,video):
        if video.ndim!=5 or video.shape[1:3]!=(4,3):raise ValueError('输入须为[B,4,3,H,W]')
        b,t,c,h,w=video.shape
        spatial=self.spatial_encoder(video.reshape(b*t,c,h,w))
        _,channels,sh,sw=spatial.shape
        temporal,encoder_aux=self.temporal_encoder(spatial.reshape(b,t,channels,sh,sw),return_aux=True)
        decoded,decoder_aux=self.temporal_decoder(temporal,return_aux=True)
        result=self.spatial_decoder(decoded.reshape(b*t,channels,sh,sw)).reshape(b,t,c,h,w)
        return result,(encoder_aux+decoder_aux)/2,{'common':0.,'individual':0.}


def build_model(variant="legacy"):
    if variant == 'spatiotemporal_only':
        return SpatiotemporalOnlyDiagnostic()
    if variant == 'spatial_only':
        return SpatialOnlyDiagnostic()
    if variant not in ("legacy", "common_fig7", "common_fig7_stage6_pair", "branches_float"):
        raise ValueError(f"未知模型版本：{variant}")
    model = base.Stage1Model()
    if variant in ("common_fig7", "common_fig7_stage6_pair", "branches_float"):
        model.common_extractor = Figure7CommonExtractor(model.common_extractor)
    if variant == 'branches_float':
        # 显式替换整个量化器：无裁剪、无训练噪声、无舍入。
        model.common_quantizer=nn.Identity()
        model.individual_quantizer=nn.Identity()
    if variant == "common_fig7_stage6_pair":
        model.individual_extractor.block = Stage6BlockPair(model.individual_extractor.block)
        model.individual_upsample.block = Stage6BlockPair(model.individual_upsample.block)
        # Stage1Model.forward会按此数量加权各模块的辅助损失均值。
        model.moe_counts = (4,2,2,4)
    return model


def transfer_float_branches_to_quantized(target, source_state):
    """保留全部已训练权重，只重新启用原固定范围量化器。

    Identity与当前UniformQuantizer都没有参数/缓冲区，故state_dict必须
    完整严格匹配。量化范围不从数据估计，也不增加传输侧信息。
    训练仍使用原裁剪加噪近似，评估使用真实舍入；不恢复旧Adam。
    """
    quantizers=(target.common_quantizer,target.individual_quantizer)
    for q in quantizers:
        if not isinstance(q,base.step09.UniformQuantizer):
            raise ValueError('目标必须使用UniformQuantizer')
        if (q.bits,q.min_value,q.max_value)!=(3,-1.,1.) or q.state_dict():
            raise ValueError('目标须为无参数的原3-bit [-1,1]量化器')
    target.load_state_dict(source_state,strict=True)
    return dict(transferred_tensors=len(source_state),new_parameter_tensors=0,
                operation='branches_float to common_fig7; all weights preserved',
                quantization_bits=3,quantization_range=[-1.,1.],
                training='clamp plus uniform noise',evaluation='clamp and round',
                optimizer='new Adam; not restored')


def transfer_spatiotemporal_codec(target, source_state):
    """只迁移四个已有编解码模块，严格核对所有新增/未知参数名称。"""
    prefixes=('spatial_encoder.','spatial_decoder.','temporal_encoder.','temporal_decoder.')
    if any(not key.startswith(prefixes) for key in source_state):
        raise ValueError('迁移源应仅包含四个编解码模块')
    loaded=target.load_state_dict(source_state,strict=False)
    new_prefixes=('common_extractor.','individual_extractor.','individual_upsample.')
    if loaded.unexpected_keys or any(not key.startswith(new_prefixes) for key in loaded.missing_keys):
        raise ValueError(f'迁移缺失或多余参数异常：{loaded}')
    return dict(transferred_tensors=len(source_state),new_parameter_tensors=len(loaded.missing_keys),
                transferred_modules=[p[:-1] for p in prefixes],new_modules=[p[:-1] for p in new_prefixes])
