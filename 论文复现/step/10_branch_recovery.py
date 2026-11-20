"""Step10：公共2D、个体3D掩码恢复模块。"""

import torch
import torch.nn as nn
import importlib

step09 = importlib.import_module("09_quantization_channel")


class CommonRecovery(nn.Module):
    """
    公共特征恢复。

    received：[B,1,C,H,W]，缺失处已置零。
    missing_mask：同形状bool张量，True表示缺失。
    """

    def __init__(self, channels=120, hidden_channels=32):
        super().__init__()

        self.channels = channels

        # 使用三层卷积预测修复值。
        # 隐藏通道32是本轮接口验证配置，后续按论文细节对齐。
        # padding=1使3×3卷积保持高、宽不变。
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )

    def forward(self, received, missing_mask):
        if (
            received.ndim != 5
            or received.shape[1] != 1
            or received.shape[2] != self.channels
        ):
            raise ValueError("公共输入必须为 [B,1,C,H,W]")

        if (
            missing_mask.shape != received.shape
            or missing_mask.dtype != torch.bool
            or missing_mask.device != received.device
        ):
            raise ValueError("mask须为同形状、同设备的bool张量")

        # 时间长度为1，移除时间维，得到Conv2d所需的NCHW。
        x = received.squeeze(1)

        prediction = self.network(x)

        # 补回时间维，与输入和mask形状一致。
        prediction = prediction.unsqueeze(1)

        # 只在缺失位置使用预测值。
        return received + prediction * missing_mask.to(received.dtype)

class IndividualRecovery(nn.Module):
    """
    个体特征恢复。

    received：[B,T,C,H,W]，缺失处已置零。
    missing_mask：同形状bool张量，True表示缺失。
    """

    def __init__(self, channels=144, hidden_channels=32):
        super().__init__()

        self.channels = channels

        # 3×3×3卷积同时覆盖时间、高和宽。
        # padding=1保持各维尺寸不变。
        self.network = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels, 3, padding=1),
        )

    def forward(self, received, missing_mask):
        if received.ndim != 5 or received.shape[2] != self.channels:
            raise ValueError("个体输入必须为 [B,T,C,H,W]")

        if (
            missing_mask.shape != received.shape
            or missing_mask.dtype != torch.bool
            or missing_mask.device != received.device
        ):
            raise ValueError("mask须为同形状、同设备的bool张量")

        # BTCHW → BCTHW，Conv3d要求通道在第二维。
        x = received.permute(0, 2, 1, 3, 4).contiguous()

        prediction = self.network(x)

        # 恢复BTCHW，与缺失mask对齐。
        prediction = prediction.permute(0, 2, 1, 3, 4).contiguous()

        return received + prediction * missing_mask.to(received.dtype)

def check_recovery(name, model, shape):
    original = torch.randn(shape)

    # 本次是模块检查，用随机元素缺失构造测试输入。
    missing_mask = torch.rand(shape) < 0.25
    received = original.masked_fill(missing_mask, 0)

    recovered = model(received, missing_mask)

    print(f"{name}输入：", tuple(received.shape))
    print(f"{name}输出：", tuple(recovered.shape))

    assert recovered.shape == original.shape
    assert torch.isfinite(recovered).all()

    # 已收到的位置必须完全保持原值。
    assert torch.equal(
        recovered[~missing_mask],
        received[~missing_mask],
    )

    # 没有缺失时，整个模块应当等价于恒等映射。
    with torch.no_grad():
        no_loss_mask = torch.zeros_like(missing_mask)
        unchanged = model(original, no_loss_mask)
        assert torch.equal(unchanged, original)

        # 全丢失时也应能正常计算，但不能据此保证恢复正确。
        all_loss_mask = torch.ones_like(missing_mask)
        all_lost_output = model(
            torch.zeros_like(original),
            all_loss_mask,
        )
        assert torch.isfinite(all_lost_output).all()

    # 用原始特征作为目标，检查恢复网络的梯度。
    loss = (recovered - original).square().mean()
    loss.backward()

    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{parameter_name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{parameter_name} 梯度存在异常")

    print(f"{name}掩码与梯度检查通过")

def check_channel_recovery():
    """检查量化、交织、整包丢失、逆交织和恢复的衔接。"""
    torch.manual_seed(42)

    quantizer = step09.UniformQuantizer(
        bits=3,
        min_value=-1.0,
        max_value=1.0,
    )
    quantizer.eval()

    branches = [
        (
            "公共分支",
            (1, 1, 120, 8, 8),
            CommonRecovery(channels=120),
        ),
        (
            "个体分支",
            (1, 4, 144, 4, 4),
            IndividualRecovery(channels=144),
        ),
    ]

    with torch.no_grad():
        for branch_index, (name, shape, recovery) in enumerate(branches):
            recovery.eval()

            # 使用范围内的示例特征检查接口。
            features = torch.rand(shape) * 2 - 1
            sent = quantizer(features)

            interleaver = step09.SemanticInterleaver(
                elements_per_sample=sent[0].numel(),
                seed=42 + branch_index,
            )

            shuffled = interleaver(sent)

            for p in (0.0, 0.25, 1.0):
                # 通道模拟在交织之后执行。
                received_shuffled, mask_shuffled, dropped = (
                    step09.transmit_packets(
                        shuffled,
                        p=p,
                        packet_size=16,
                    )
                )

                # 特征和mask必须使用相同的逆排列。
                received = interleaver.deinterleave(
                    received_shuffled
                )
                missing_mask = interleaver.deinterleave(
                    mask_shuffled
                )

                # 卷积处理原布局下的邻域。
                recovered = recovery(received, missing_mask)

                assert recovered.shape == sent.shape
                assert torch.isfinite(recovered).all()

                # 恢复模块只能修改缺失位置。
                assert torch.equal(
                    recovered[~missing_mask],
                    received[~missing_mask],
                )

                if p == 0.0:
                    # 无丢包时，量化发送特征应完整保留。
                    assert torch.equal(recovered, sent)

                # 本检查以“量化后发送特征”为目标，
                # 单独观察通道与恢复造成的变化。
                zero_fill_mse = (
                    received - sent
                ).square().mean().item()

                recovery_mse = (
                    recovered - sent
                ).square().mean().item()

                actual_loss = dropped.float().mean().item()

                print(
                    f"{name}，设定丢包={p:.0%}，"
                    f"实际丢包={actual_loss:.1%}"
                )
                print(
                    f"  置零 MSE={zero_fill_mse:.6f}，"
                    f"恢复后 MSE={recovery_mse:.6f}"
                )

    print("交织通道与双分支恢复联合检查通过")

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    check_recovery(
        name="公共恢复",
        model=CommonRecovery(channels=120),
        shape=(1, 1, 120, 8, 8),
    )

    check_recovery(
        name="个体恢复",
        model=IndividualRecovery(channels=144),
        shape=(1, 4, 144, 4, 4),
    )
    check_channel_recovery()

if __name__ == "__main__":
    main()
