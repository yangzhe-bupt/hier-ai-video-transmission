# ======================================================================
# 文件：shifted_window.py
# 位置：practice/models/swin
# 功能：生成边界掩码，执行循环移位、带掩码窗口处理和空间还原。
# 类与函数汇总：
#   - build_shift_mask：标记循环移位后同一窗口内不应互相注意的Token对。
#   - check_shift_mask：检查掩码形状与基本屏蔽行为。
#   - check_masked_attention：检查多个样本的窗口是否正确复用掩码。
#   - ShiftedWindowBlock：对二维特征图执行移位窗口Transformer处理。
#     方法：__init__、forward
#   - check_shifted_window_block：检查输出形状、梯度以及移位后的坐标还原。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/swin/shifted_window.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch.nn as nn
import torch
from window_attention import WindowAttention
from spatial_window_block import partition_windows, reverse_windows
from window_transformer_block import WindowTransformerBlock
# -----------------1.生成移位窗口掩码-----------------------
def build_shift_mask(
    height,
    width,
    window_size=4,
    shift_size=2,
    device="cpu"
):
    """
    标记循环移位后同一窗口内不应互相注意的Token对。

    参数：
        height：特征图高度
        width：特征图宽度
        window_size：窗口边长
        shift_size：移位距离，满足0 < shift_size < window_size
        device：掩码所在设备

    返回：
        mask：浮点张量，[窗口数, Token数, Token数]
            0：允许注意
            -inf：禁止注意，在Softmax后对应权重0
    """
    if height <= 0 or width <= 0:
        raise ValueError("高、宽必须大于0")

    if window_size <= 0:
        raise ValueError("窗口边长必须大于0")

    if height % window_size or width % window_size:
        raise ValueError("高、宽必须能被窗口边长整除")

    if not 0 < shift_size < window_size:
        raise ValueError("移位距离必须大于0且小于窗口边长")

    # ---- 第1步：创建区域编号图 ----
    # labels: [1, 1, H, W]，每个位置存一个区域编号
    labels = torch.zeros(
        1,1,height,width,
        dtype=torch.long,device=device
    )

    # 行方向切成 3 段：
    #   [0, H-M)        → 上方"非边界"区域
    #   [H-M, H-shift)  → 中间过渡带
    #   [H-shift, H)    → 下方边界区域
    row_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None)
    )
    # 列方向同理，行列组合共 3×3 = 9 个区域
    col_slices = row_slices
    region_id = 0

    # ---- 第2步：给 9 个区域分配不同编号 ----
    # 每个 (rows, cols) 组合是一个矩形区域，赋一个唯一 ID
    for rows in row_slices:
        for cols in col_slices:
            labels[:, :, rows, cols] = region_id
            region_id += 1

    # ---- 第3步：把编号图按窗口划分 ----
    label_windows = partition_windows(
        labels, window_size
    ).squeeze(-1)   # [窗口数, M²]

    # ---- 第4步：窗口内两两比较编号 ----
    # 原本属于不同区域，是移位拼进来的假邻居
    different_region = (
        label_windows.unsqueeze(2)
        != label_windows.unsqueeze(1)
    )

     # ---- 第5步：构造浮点 mask ----
    # 先全 0（允许注意），再把"不同区域"的位置填 -inf
    # 用 -inf 而不是 1/0，是因为后面直接和 scores 相加或
    # masked_fill，softmax(-inf)=0，正好实现"权重为0"
    mask = torch.zeros(
        different_region.shape,
        dtype=torch.float32,
        device=device
    )

    return mask.masked_fill(different_region, float("-inf"))

# -----------------2.检查掩码-----------------------
def check_shift_mask():
    """
    检查掩码形状与基本屏蔽行为。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    mask = build_shift_mask(
        height=8,
        width=8,
        window_size=4,
        shift_size=2
    )

    # 8×8特征图包含4个窗口，每窗16个Token
    assert mask.shape == (4, 16, 16)

    # Token始终可以注意自身
    assert (mask.diagonal(dim1=-2, dim2=-1) == 0).all().item()

    # 两个Token是否跨边界，与比较方向无关
    assert torch.equal(mask, mask.transpose(-2, -1))

    # 左上窗口不涉及循环拼接边界
    assert (mask[0] == 0).all().item()

    # 其他边界窗口应存在禁止注意的位置
    assert torch.isneginf(mask).any().item()

    # 模拟将掩码加到全零注意力分数上，再做Softmax
    weights = mask.softmax(dim=-1)
    blocked = torch.isneginf(mask)

    assert torch.isfinite(weights).all().item()
    assert (weights[blocked] == 0).all().item()
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1))
    )

    print("移位窗口掩码：", mask.shape)
    print("掩码形状、边界屏蔽与归一化检查通过")

# -----------------检查掩码接入注意力-----------------------
def check_masked_attention():
    """
    检查多个样本的窗口是否正确复用掩码。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    mask = build_shift_mask(8, 8, window_size=4, shift_size=2)

    model = WindowAttention(
        dim=24,
        num_heads=3,
        window_size=4
    )

    # 2张特征图，每张4个窗口，共8个窗口
    windows = torch.randn(8, 16, 24, requires_grad=True)

    output, attention = model(windows, mask=mask)

    assert output.shape == (8, 16, 24)
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(attention).all().item()

    # 还原样本与窗口维度，检查所有头的屏蔽位置
    grouped_attention = attention.reshape(2, 4, 3, 16, 16)

    blocked = torch.isneginf(mask)
    blocked = blocked.unsqueeze(0).unsqueeze(2)
    blocked = blocked.expand_as(grouped_attention)

    assert (grouped_attention[blocked] == 0).all().item()

    output.square().mean().backward()

    assert windows.grad is not None
    assert torch.isfinite(windows.grad).all().item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("带掩码注意力输出：", output.shape)
    print("多样本掩码屏蔽与梯度检查通过")

# -----------------移位窗口处理模块-----------------------
class ShiftedWindowBlock(nn.Module):
    """
    对二维特征图执行移位窗口Transformer处理。

    参数：
        dim：特征通道数
        num_heads：注意力头数
        window_size：窗口边长
        shift_size：循环移位距离
        mlp_ratio：MLP隐藏层扩大倍数

    输入与返回：
        [N,dim,H,W]，高宽必须能被窗口边长整除
    """
    def __init__(
        self,
        dim=24,
        num_heads=3,
        window_size=4,
        shift_size=2,
        mlp_ratio=4
    ):
        super().__init__()

        if not 0 < shift_size < window_size:
            raise ValueError("移位距离必须大于0且小于窗口边长")

        self.window_size = window_size
        self.shift_size = shift_size

        self.block = WindowTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio
        )

    def forward(self,features):
        if features.ndim != 4:
            raise ValueError("输入必须为[N,C,H,W]")

        N,C,H,W = features.shape

        if H <= self.window_size or W <= self.window_size:
            raise ValueError("当前移位模块要求高宽都大于窗口边长")

        # ---- 1. 生成移位掩码 ----
        mask = build_shift_mask(
            height=H,
            width=W,
            window_size=self.window_size,
            shift_size=self.shift_size,
            device=features.device
        )
        # ---- 2. 负向循环移位 ----
        # 向左上滚 shift 格，让窗口边界错开，跨窗口 token 有机会进同一窗口。
        # 方向必须和 build_shift_mask 里的 3×3 切片方向配套：
        # 负向移位配 [0,H-M), [H-M,H-shift), [H-shift,H) 切片。
        shifted = torch.roll(
            features,
            shifts=(-self.shift_size, -self.shift_size),
            dims=(2, 3)
        )

        # ---- 3. 划分窗口 ----
        # [N,C,H,W] → [N*nH*nW, M², C]
        windows = partition_windows(
            shifted, self.window_size
        )

        # ---- 4. 带掩码的窗口 Transformer ----
        windows = self.block(windows, mask=mask)

        # ---- 5. 还原特征图 ----
        # [N*nH*nW, M², C] → [N,C,H,W]
        shifted_output = reverse_windows(
            windows,
            window_size=self.window_size,
            batch_size=N,
            height=H,
            width=W
        )

        # ---- 6. 正向移回，抵消第2步 ----
        # shifts 必须和第2步互为相反数，dims 相同，
        output = torch.roll(
            shifted_output,
            shifts=(self.shift_size, self.shift_size),
            dims=(2, 3)
        )

        return output


# -----------------检查完整移位窗口模块-----------------------
def check_shifted_window_block():
    """
    检查输出形状、梯度以及移位后的坐标还原。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = ShiftedWindowBlock()

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

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("移位窗口输出：", output.shape)
    print("移位窗口形状与梯度检查通过")

    # 将更新分支置零，检查划窗、还原、移位是否保持空间对应
    with torch.no_grad():
        model.block.attention.proj.weight.zero_()
        model.block.attention.proj.bias.zero_()
        model.block.mlp[-1].weight.zero_()
        model.block.mlp[-1].bias.zero_()

        inputs = features.detach()
        restored = model(inputs)

    assert torch.equal(restored, inputs)
    print("移位与空间还原一致性检查通过")

# -----------------4.主程序-----------------------
def main():
    """运行移位窗口掩码检查。"""
    check_shift_mask()
    check_masked_attention()
    check_shifted_window_block()

if __name__ == "__main__":
    main()