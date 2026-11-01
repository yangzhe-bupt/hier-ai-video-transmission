# ======================================================================
# 文件：window_transformer_block.py
# 位置：practice/models/swin
# 功能：组合预归一化、窗口注意力、MLP及残差连接；支持向注意力传入掩码。
# 类与函数汇总：
#   - WindowTransformerBlock：使用注意力和MLP更新窗口Token
#     方法：__init__、forward
#   - check_window_transformer_block：检查输出形状、梯度以及残差连接的行为。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/swin/window_transformer_block.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn

from window_attention import WindowAttention

# -----------------1.非移位窗口Transformer块-----------------------
class WindowTransformerBlock(nn.Module):
    """
    使用注意力和MLP更新窗口Token

    参数：
        dim：Token通道数
        num_heads：注意力头数
        window_size：窗口边长
        mlp_ratio：MLP隐藏层相对输入通道数的扩大倍数

    输入：
        windows：[窗口总数, window_size², dim]

    返回：
        x：与输入形状相同的窗口特征
    """
    def __init__(
            self,
            dim=24,
            num_heads=3,
            window_size=4,
            mlp_ratio=4
    ):
        super().__init__()

        # MLP中间层维度，惯例放大4倍（24 → 96）
        hidden_dim = int(dim*mlp_ratio)

        if hidden_dim <= 0:
            raise ValueError("MLP隐藏层通道数必须大于0")

        # ---- 子层1 的归一化 ----
        self.norm1 = nn.LayerNorm(dim)

        # ---- 子层1 的主体：窗口内多头注意力（含相对位置偏置）----
        # 输入输出都是 [窗口总数, N, dim]，形状不变，方便接残差
        self.attention = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size
        )

        # ---- 子层2 的归一化 ----
        self.norm2 = nn.LayerNorm(dim)

        # ---- 子层2 的主体：MLP ----
        # 对每个token独立做 24 → 96 → 24 的两层线性 + GELU。
        # 注意：nn.Linear 作用在最后一维，不跨token，等价于同一组
        # 权重被复制N份并行跑。它负责token内部的非线性特征变换，
        # 和注意力的「token之间交换信息」互补。
        self.mlp = nn.Sequential(
            nn.Linear(dim,hidden_dim), #升维
            nn.GELU(), #非线性
            nn.Linear(hidden_dim,dim) #降维
        )

    def forward(self,windows,mask=None):
        # ============ 子层1：LN → 注意力 → 残差 ============
        attention_output, _ = self.attention(
            self.norm1(windows),
            mask=mask
        )
        # 3) 残差连接：把子层输入直接加到子层输出上。
        x = windows + attention_output

        # ============ 子层2：LN → MLP → 残差 ============
        # 4) 对子层1的输出再做一次LayerNorm
        x = x + self.mlp(self.norm2(x))
        # 输出形状与输入完全相同，因此可以继续堆叠下一个block
        return x

# -----------------2.检查窗口Transformer块-----------------------
def check_window_transformer_block():
    """
    检查输出形状、梯度以及残差连接的行为。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = WindowTransformerBlock(
        dim=24,
        num_heads=3,
        window_size=4,
        mlp_ratio=4
    )

    windows = torch.randn(
        256, 16, 24,
        requires_grad=True
    )

    output = model(windows)

    assert output.shape == windows.shape
    assert torch.isfinite(output).all().item()

    loss = output.square().mean()
    loss.backward()

    assert windows.grad is not None
    assert torch.isfinite(windows.grad).all().item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("窗口块输出：", output.shape)
    print("窗口块形状与梯度检查通过")

    # 单独创建检查模型，不影响上面的模型
    residual_model = WindowTransformerBlock()

    # 将两条更新分支的最后投影置零，
    # 此时整个块应仅保留残差路径，输出等于输入
    with torch.no_grad():
        residual_model.attention.proj.weight.zero_()
        residual_model.attention.proj.bias.zero_()

        residual_model.mlp[-1].weight.zero_()
        residual_model.mlp[-1].bias.zero_()

        inputs = windows.detach()
        actual = residual_model(inputs)

    assert torch.equal(actual, inputs)
    print("残差连接检查通过")


# -----------------3.主程序-----------------------
def main():
    check_window_transformer_block()


if __name__ == "__main__":
    main()