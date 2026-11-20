"""阶段II学习模型：冻结发送端、整包擦除、双分支恢复、接收端解码。

实验选择：冻结发送端后使用评估舍入量化；恢复目标为发送的量化特征。
论文式17目标是否在量化前尚未完全确认，此选择须在报告中注明。
"""
import importlib
import torch
from torch import nn

variants = importlib.import_module('12_model_variants')
recovery = importlib.import_module('10_branch_recovery')
base = variants.base


class Stage2Model(nn.Module):
    frozen_names = ('spatial_encoder','temporal_encoder','common_extractor',
                    'individual_extractor','common_quantizer','individual_quantizer')

    def __init__(self, stage1, packet_size=16, recovery_enabled=True):
        super().__init__()
        if packet_size < 1:
            raise ValueError('packet_size必须为正')
        self.codec = stage1
        self.packet_size = packet_size
        self.recovery_enabled = recovery_enabled
        self.common_recovery = recovery.CommonRecovery(120)
        self.individual_recovery = recovery.IndividualRecovery(144)
        # 保留相同初始化过程和state_dict布局，保证两种实验其余随机状态一致。
        # 无恢复对照不会执行这些网络，也不将其参数交给优化器。
        if not recovery_enabled:
            self.common_recovery.requires_grad_(False)
            self.individual_recovery.requires_grad_(False)
        self.interleavers = nn.ModuleList([
            base.step09.SemanticInterleaver(120*8*8,42),
            base.step09.SemanticInterleaver(4*144*4*4,43)])
        for name in self.frozen_names:
            getattr(self.codec,name).requires_grad_(False).eval()
        # 钩子保持原编解码路径。只包装量化后的输出，不改原权重。
        self._handles = [module.register_forward_hook(lambda m,i,o,b=b:self._receive(b,o))
                         for b,module in enumerate((self.codec.common_quantizer,self.codec.individual_quantizer))]

    def train(self, mode=True):
        super().train(mode)
        for name in self.frozen_names:
            getattr(self.codec,name).eval()
        return self

    def _receive(self, branch, clean):
        interleaver = self.interleavers[branch]
        n = clean[0].numel()
        packets = (n+self.packet_size-1)//self.packet_size
        masks, packet_masks = [], []
        for index, (p, seed) in enumerate(zip(self._probabilities,self._seeds)):
            generator = torch.Generator().manual_seed(int(seed)+branch*1000003)
            missing = torch.rand(packets,generator=generator) < float(p)
            packet_masks.append(missing)
            masks.append(missing.repeat_interleave(self.packet_size)[:n])
        mask = torch.stack(masks).reshape(clean.shape).to(clean.device)
        received = interleaver.deinterleave(interleaver(clean).masked_fill(mask,0))
        missing_mask = interleaver.deinterleave(mask)
        module = self.common_recovery if branch == 0 else self.individual_recovery
        restored = module(received,missing_mask) if self.recovery_enabled else received
        # 已收到位置保持原样，恢复仅作用于缺失位置。
        if not torch.equal(restored[~missing_mask],clean[~missing_mask]):
            raise RuntimeError('恢复模块改变了已收到的特征')
        self.branches.append(dict(clean=clean.detach(), received=received, restored=restored,
                                  missing_mask=missing_mask, packet_masks=torch.stack(packet_masks)))
        return restored

    def forward(self, video, probabilities, seeds):
        if video.shape[1:] != (4,3,256,256):
            raise ValueError('当前阶段II支持[B,4,3,256,256]')
        if len(probabilities)!=len(video) or len(seeds)!=len(video):
            raise ValueError('每个GOP需要独立概率和通道种子')
        if any(not 0<=float(p)<=1 for p in probabilities):
            raise ValueError('丢包概率必须在[0,1]')
        self._probabilities, self._seeds = probabilities, seeds
        self.branches = []
        prediction, auxiliary, saturation = self.codec(video)
        # 整分支MSE；非缺失位置为零误差。不是仅在缺失元素上归一化。
        feature_loss = sum((r['restored']-r['clean']).square().mean() for r in self.branches)
        return prediction, auxiliary, feature_loss, saturation
