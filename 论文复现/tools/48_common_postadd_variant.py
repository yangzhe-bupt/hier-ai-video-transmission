"""独立图7激活位置变体：只移动最后一个GELU，不改旧模型定义。"""
import importlib
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
codec = importlib.import_module('40_residual_full_codec')
variants = importlib.import_module('12_model_variants')


class PostAddCommonExtractor(variants.Figure7CommonExtractor):
    def forward(self, features):
        if features.ndim != 5 or features.shape[1:3] != (self.gop_size, self.channels):
            raise ValueError('需要[B,4,C,H,W]')
        x = features.permute(0, 2, 1, 3, 4).contiguous()
        x = self.channel_mixing(self.activation(self.temporal_compression(x)))
        if x.shape[2] != 1:
            raise ValueError('公共分支时间维应为1')
        s = x.squeeze(2)
        # 保留原模块/state_dict名称；refinement[3]移到Add之后调用。
        residual = self.refinement[2](self.refinement[1](self.refinement[0](s)))
        return self.refinement[3](s + residual).unsqueeze(1)


def build_model(name):
    postadd = name.endswith('_postadd')
    base_name = name[:-8] if postadd else name
    model = codec.factory(base_name) if base_name in codec.NAMES else codec.original_build(base_name)
    if postadd:
        if not isinstance(getattr(model, 'common_extractor', None), variants.Figure7CommonExtractor):
            raise ValueError('该模型没有兼容的图7公共分支')
        # 不重建层、不额外消耗初始化RNG；只有前向语义改变。
        model.common_extractor.__class__ = PostAddCommonExtractor
    return model


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    old = build_model('residual_branches_moe_3bit')
    torch.manual_seed(42)
    new = build_model('residual_branches_moe_3bit_postadd')
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(1, 4, 120, 8, 8, requires_grad=True)
    q = new.common_extractor
    s = q.channel_mixing(q.activation(q.temporal_compression(x.permute(0,2,1,3,4)))).squeeze(2)
    residual = q.refinement[2](q.refinement[1](q.refinement[0](s)))
    torch.testing.assert_close(q(x), nn.functional.gelu(s+residual).unsqueeze(1))
    torch.testing.assert_close(old.common_extractor(x), (s+nn.functional.gelu(residual)).unsqueeze(1))
    assert not torch.allclose(q(x), old.common_extractor(x))
    q(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
    new.eval()
    with torch.no_grad():
        y, aux, _ = new(torch.rand(1,4,3,256,256))
    assert y.shape == (1,4,3,256,256) and torch.isfinite(y).all() and torch.isfinite(aux)
    print('CPU检查通过：相同初始参数/严格加载/新旧公式/梯度/完整前向；参数更新0次')


if __name__ == '__main__':
    self_test()
