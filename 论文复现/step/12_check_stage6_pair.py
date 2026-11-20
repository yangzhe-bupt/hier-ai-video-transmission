"""Stage6结构验证：检查单窗口规则、多窗口掩码、12路由器梯度。

不创建优化器，不更新参数；结果保存到独立检查目录。
"""
import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import torch

variants = importlib.import_module('12_model_variants')
base = variants.base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent.parent / 'outputs/stage6_checks')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = variants.build_model('common_fig7_stage6_pair').to(args.device)
    pair = model.individual_extractor.block
    # 单窗口：第二块应等同于普通窗口块，不能被移位mask拆成小区域。
    x = torch.randn(1,4,144,4,4, device=args.device)
    with torch.no_grad():
        y = pair(x)
        assert pair.second.shift_size == (0,0,0)
        torch.testing.assert_close(y, pair.second(pair.first(x)))
        # 8×8包含四个空间窗口，应启用移位；切回4×4还应禁用。
        z = pair(torch.randn(1,4,144,8,8, device=args.device))
        assert z.shape == (1,4,144,8,8) and torch.isfinite(z).all()
        assert pair.second.shift_size == (0,2,2)
        pair(x)
        assert pair.second.shift_size == (0,0,0)
        a = model.individual_extractor.block
        _, aux = a(x, return_aux=True)
        first, aux1 = a.first(x, return_aux=True)
        _, aux2 = a.second(first, return_aux=True)
        torch.testing.assert_close(aux, (aux1+aux2)/2)
    # 辅助损失汇总应等于12个路由器的损失均值，使用hook直接获取。
    router_losses = []
    routers = [m for m in model.modules() if isinstance(m, base.step11.TopKRouter)]
    handles = [r.register_forward_hook(lambda m,i,o: router_losses.append(o[2])) for r in routers]
    video = torch.rand(1,4,3,256,256, device=args.device)
    prediction, auxiliary, _ = model(video)
    assert len(routers) == len(router_losses) == 12
    torch.testing.assert_close(auxiliary, torch.stack(router_losses).mean())
    for handle in handles:
        handle.remove()
    grads = torch.autograd.grad(auxiliary, [r.gate.weight for r in routers], retain_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    loss = (prediction-video).square().mean()+.01*auxiliary
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert '.experts.' in name, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
    report = {'device': args.device, 'torch': torch.__version__, 'optimizer_updates': 0,
        'variant': 'common_fig7_stage6_pair', 'moe_counts': model.moe_counts,
        'parameters': sum(p.numel() for p in model.parameters()),
        'single_window_shift_disabled': True, 'multi_window_shift_enabled': True,
        'auxiliary_equals_12_router_mean': True, 'all_router_gradients_finite': True,
        'input_shape': list(video.shape), 'output_shape': list(prediction.shape),
        'auxiliary_loss': auxiliary.item()}
    out = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'检查记录：{out}')


if __name__ == '__main__':
    main()
