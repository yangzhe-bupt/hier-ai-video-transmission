# ======================================================================
# 文件：spatial_window_block.py
# 位置：practice/models/swin
# 功能：划分与还原二维特征窗口，并执行非移位窗口Transformer。
# 类与函数汇总：
#   - partition_windows：将二维特征图划分为互不重叠的窗口。
#   - reverse_windows：将窗口Token恢复为二维特征图。
#   - SpatialWindowBlock：对二维特征图执行非移位窗口Transformer处理。
#     方法：__init__、forward
#   - check_spatial_window_block：检查窗口划分顺序、还原一致性、模块输出和梯度。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/swin/spatial_window_block.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn

from window_transformer_block import WindowTransformerBlock

# -----------------1.划分窗口-----------------------
def partition_windows(features, window_size):
    """
    将二维特征图划分为互不重叠的窗口。

    参数：
        features：特征图，[N,C,H,W]
        window_size：窗口边长

    返回：
        windows：[N*窗口行数*窗口列数, window_size², C]
    """
    if features.ndim != 4:
        raise ValueError("输入必须为[N,C,H,W]")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("窗口边长必须为正整数")

    N,C,H,W = features.shape

    if H % window_size != 0 or W % window_size != 0:
        raise ValueError("高、宽必须能被窗口边长整除")

    #[N,H,W,C]
    x = features.permute(0,2,3,1)

    # 拆分为特征图,窗口行、窗内行、窗口列、窗内列，通道数
    x = x.reshape(
        N,H//window_size,window_size,W//window_size,window_size,C
    )
    # 将同一个窗口的行列放在一起
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window_size * window_size, C)

# -----------------2.还原窗口-----------------------
def reverse_windows(windows, window_size, batch_size, height, width):
    """
    将窗口Token恢复为二维特征图。

    参数：
        windows：窗口张量，[窗口总数, window_size², C]
        window_size：窗口边长
        batch_size：原特征图的N
        height：原特征图高度
        width：原特征图宽度

    返回：
        features：[N,C,H,W]
    """
    if windows.ndim != 3:
        raise ValueError("窗口输入必须为三维张量")

    if window_size <= 0 or batch_size <= 0:
        raise ValueError("窗口边长与批量大小必须大于0")

    if height <= 0 or width <= 0:
        raise ValueError("高、宽必须大于0")

    if height % window_size != 0 or width % window_size != 0:
        raise ValueError("高、宽必须能被窗口边长整除")
    
    # 每张图有 (H/M) × (W/M) 个窗口，共 batch_size 张图
    C = windows.shape[-1]
    expected_count = (
        batch_size*(height//window_size)*(width//window_size)
    )
    
    if windows.shape[:2] != (
        expected_count, window_size * window_size
    ):
        raise ValueError("窗口数量或每窗Token数不匹配")

    # ---- 维度还原（partition 的逆过程）----
    x = windows.reshape(
        batch_size,
        height // window_size,   # nH：窗口行数
        width // window_size,    # nW：窗口列数
        window_size,             # M：窗内行数
        window_size,             # M：窗内列数
        C
    )
    x = x.permute(0, 1, 3, 2, 4, 5)
    # [N,nH,M,nW,M,C] → [N,H,W,C]
    # 因为 H = nH*M、W = nW*M，reshape 会自动按行优先把「窗口行 × 窗内行」拼成完整 H，「窗口列 × 窗内列」拼成完整 W
    x = x.reshape(batch_size, height, width, C)
    return x.permute(0, 3, 1, 2).contiguous()

# -----------------3.二维特征窗口处理模块-----------------------
class SpatialWindowBlock(nn.Module):
    """
    对二维特征图执行非移位窗口Transformer处理。

    参数：
        dim：特征通道数
        num_heads：注意力头数
        window_size：窗口边长
        mlp_ratio：MLP隐藏层扩大倍数

    输入与返回：
        [N,dim,H,W]，处理前后形状相同
    """
    def __init__(
        self,
        dim=24,
        num_heads=3,
        window_size=4,
        mlp_ratio=4
    ):
        super().__init__()

        self.window_size = window_size

        self.block = WindowTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio
        )

    def forward(self,features):
        #划分窗口
        windows = partition_windows(features,self.window_size)
        N,C,H,W = features.shape

        # 注意力和MLP只在各个窗口内处理
        windows = self.block(windows)

        return reverse_windows(
            windows,
            window_size=self.window_size,
            batch_size=N,
            height=H,
            width=W
        )

# -----------------4.检查模块-----------------------
def check_spatial_window_block():
    """
    检查窗口划分顺序、还原一致性、模块输出和梯度。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    # 使用非方形特征图，检查高、宽是否正确还原
    features = torch.randn(
        2, 24, 32, 48,
        requires_grad=True
    )

    windows = partition_windows(features, window_size=4)

    # 第一个窗口应对应第一个样本左上角4×4区域
    expected = features[0, :, :4, :4]
    expected = expected.permute(1, 2, 0).reshape(16, 24)

    assert torch.equal(windows[0], expected)

    restored = reverse_windows(
        windows,
        window_size=4,
        batch_size=2,
        height=32,
        width=48
    )

    assert torch.equal(restored, features)
    print("窗口空间顺序与还原检查通过")

    model = SpatialWindowBlock()
    output = model(features)

    assert output.shape == features.shape
    assert torch.isfinite(output).all().item()

    output.square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all().item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("二维特征输出：", output.shape)
    print("二维窗口模块与梯度检查通过")


# -----------------5.主程序-----------------------
def main():
    """检查二维特征图的窗口处理流程。"""
    check_spatial_window_block()


if __name__ == "__main__":
    main()