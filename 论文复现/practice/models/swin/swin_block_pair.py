# ======================================================================
# 文件：swin_block_pair.py
# 位置：practice/models/swin
# 功能：串联非移位与移位窗口块，构建基础二维Swin块对；尚未接入视频编码器。
# 类与函数汇总：
#   - SwinBlockPair：串联一个非移位窗口块和一个移位窗口块。
#     方法：__init__、forward
#   - check_swin_block_pair：检查串联后的形状、数值及全部参数的梯度。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/swin/swin_block_pair.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn

from spatial_window_block import SpatialWindowBlock
from shifted_window import ShiftedWindowBlock

# -----------------1.基础Swin块对-----------------------
class SwinBlockPair(nn.Module):
    """
    串联一个非移位窗口块和一个移位窗口块。

    参数：
        dim：输入与输出特征通道数
        num_heads：注意力头数
        window_size：窗口边长
        mlp_ratio：MLP隐藏层扩大倍数

    输入：
        features：[N,dim,H,W]

    返回：
        output：与输入形状相同的特征图
    """
    def __init__(
        self,
        dim=24,
        num_heads=3,
        window_size=4,
        mlp_ratio=4
    ):
        super().__init__()

        if window_size < 2:
            raise ValueError("窗口边长至少为2")

        self.regular_block = SpatialWindowBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio
        )

        self.shifted_block = ShiftedWindowBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size//2,
            mlp_ratio=mlp_ratio
        )

    def forward(self,features):
        # 固定窗口内的信息交互
        x = self.regular_block(features)

        # 改变窗口划分，进一步交换信息
        output = self.shifted_block(x)

        return output

# -----------------2.检查Swin块对-----------------------
def check_swin_block_pair():
    """
    检查串联后的形状、数值及全部参数的梯度。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = SwinBlockPair(
        dim=24,
        num_heads=3,
        window_size=4
    )

    features = torch.randn(
        2, 24, 32, 48,
        requires_grad=True
    )

    output = model(features)

    assert output.shape == features.shape
    assert torch.isfinite(output).all().item()

    output.square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all().item()

    # 同时检查非移位块和移位块的参数
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("Swin块对输入：", features.shape)
    print("Swin块对输出：", output.shape)
    print("非移位与移位窗口串联及梯度检查通过")


# -----------------3.主程序-----------------------
def main():
    """运行基础Swin块对检查。"""
    check_swin_block_pair()


if __name__ == "__main__":
    main()