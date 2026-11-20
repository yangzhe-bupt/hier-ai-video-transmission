# ======================================================================
# 文件：window_attention.py
# 位置：practice/models/swin
# 功能：窗口多头自注意力：包含QKV投影、相对位置偏置和可选移位掩码。
# 类与函数汇总：
#   - WindowQKV：对每个窗口的Token生成Q、K、V,并拆分多个注意力头
#     方法：__init__、forward
#   - check_window_qkv：检查多头Q、K、V的形状、数值和梯度。
#   - RelativePositionBias：为方形窗口生成可学习的相对位置偏置
#     方法：__init__、forward
#   - check_relative_position_bias：检查位置索引、偏置形状和梯度。
#   - WindowAttention：在每个窗口内部计算多头注意力
#     方法：__init__、forward
#   - check_window_attention：检查窗口注意力的形状、权重归一化和梯度。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/swin/window_attention.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn

# -----------------1.生成多头Q、K、V-----------------------
class WindowQKV(nn.Module):
    """
    对每个窗口的Token生成Q、K、V,并拆分多个注意力头

    参数：
        dim:输入Token的通道数
        num_heads:注意力头数,必须能整除dim

    输入：
        windows:[窗口总数,每窗Token数,dim]

    返回：
        q、k、v:每个张量的形状为[窗口总数,num_heads,每窗Token数,每头通道数]
    """

    def __init__(self, dim,num_heads):
        super().__init__()

        if dim <= 0 or num_heads <= 0:
            raise ValueError("通道数和注意力头数必须大于0")

        if dim % num_heads != 0:
            raise ValueError("通道数必须能被注意力头数整除")   

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads

        # 一次线性映射，同时生成Q、K、V
        # 最后一维：dim → 3*dim
        self.qkv = nn.Linear(dim,3*dim)

    def forward(self,windows):
        if windows.ndim != 3:
            raise ValueError("输入必须为[窗口总数, Token数, 通道数]")

        num_windows,num_tokens,channels = windows.shape

        if channels != self.dim:
            raise ValueError("输入通道数与模型设置不一致")

        # [256,16,24] → [256,16,72]
        qkv = self.qkv(windows)

        # 将72拆成：3组 × 3个头 × 每头8个通道
        # [256,16,72] → [256,16,3,3,8]
        qkv = qkv.reshape(
            num_windows,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim
        )

         # [窗口数,Token数,QKV,头数,头维度] → [QKV,窗口数,头数,Token数,头维度]
         #[3,256,3,16,8]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 沿最前面的QKV维度拆成三个张量
        q, k, v = qkv.unbind(dim=0)
        return q, k, v

# -----------------2.检查Q、K、V-----------------------
def check_window_qkv():
    """
    检查多头Q、K、V的形状、数值和梯度。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = WindowQKV(dim=24, num_heads=3)

    # 模拟256个窗口，每窗16个Token，每个Token有24个通道
    windows = torch.randn(
        256, 16, 24,
        requires_grad=True
    )

    q, k, v = model(windows)

    expected_shape = (256, 3, 16, 8)

    for name, tensor in [("Q", q), ("K", k), ("V", v)]:
        assert tensor.shape == expected_shape
        assert torch.isfinite(tensor).all().item()
        print(f"{name}形状：", tensor.shape)

    # 检查线性投影和多头拆分能否传递梯度
    loss = q.square().mean() + k.square().mean() + v.square().mean()
    loss.backward()

    assert windows.grad is not None
    assert torch.isfinite(windows.grad).all().item()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()

    print("QKV形状与梯度检查通过")

# -----------------3.相对位置偏移-------------------------
class RelativePositionBias(nn.Module):
    """
    为方形窗口生成可学习的相对位置偏置

    参数：
        window_size：窗口边长，例如4表示4×4个Token
        num_heads：注意力头数

    输入：
        无
        位置关系由窗口大小确定

    返回：
        bias：[num_heads, Token数, Token数]
    """
    def __init__(self,window_size,num_heads):
        super().__init__()

        if window_size <= 0 or num_heads <= 0:
            raise ValueError("窗口边长和注意力头数必须大于0")
        self.window_size = window_size
        self.num_heads = num_heads

        #相对位置种类数
        num_positions = (2*window_size-1)**2

        #建立偏置表
        self.bias_table = nn.Parameter(
            torch.zeros(num_positions,num_heads)
        )

        #创建二维坐标
        rows,cols = torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij"
        )
        coords = torch.stack([rows,cols]).flatten(1)

        #计算所有 token 对的相对坐标
        relative_coords = (
            coords[:,:,None] - coords[:,None,:]
        )

        #非负平移
        relative_coords = relative_coords + window_size -1

        #二维索引压成一维
        position_index = (
            relative_coords[0]*(2*window_size-1) + relative_coords[1]
        )

        # 固定索引不是训练参数，但需要随模型移动设备
        self.register_buffer("position_index", position_index)

        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self):
        num_tokens = self.window_size **2

        #查偏置表
        bias = self.bias_table[self.position_index.reshape(-1)]
        # [Token数,Token数,num_heads]
        bias = bias.reshape(num_tokens,num_tokens,self.num_heads)

        # [num_heads,Token数,Token数]
        return bias.permute(2, 0, 1).contiguous()

# -----------------4.检查相对位置偏置-----------------------
def check_relative_position_bias():
    """
    检查位置索引、偏置形状和梯度。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = RelativePositionBias(
        window_size=4,
        num_heads=3
    )

    index = model.position_index
    bias = model()

    assert model.bias_table.shape == (49, 3)
    assert index.shape == (16, 16)
    assert bias.shape == (3, 16, 16)
    assert index.min().item() == 0
    assert index.max().item() == 48

    # 自身到自身的相对距离为(0,0)，对应索引24
    assert torch.equal(
        index.diagonal(),
        torch.full((16,), 24, dtype=torch.long)
    )

    # 逐行排列时：
    # Token0→Token1 和 Token1→Token2 都是向右一格
    assert index[0, 1].item() == index[1, 2].item()

    # Token0→Token4是向下一格，应区别于向右一格
    assert index[0, 1].item() != index[0, 4].item()

    # 同一相对位置应查到相同偏置
    assert torch.equal(bias[:, 0, 1], bias[:, 1, 2])

    bias.square().mean().backward()

    assert model.bias_table.grad is not None
    assert torch.isfinite(model.bias_table.grad).all().item()

    print("偏置表形状：", model.bias_table.shape)
    print("位置索引形状：", index.shape)
    print("相对位置偏置：", bias.shape)
    print("相对位置索引与梯度检查通过")

# -----------------5.窗口内多头注意力-----------------------
class WindowAttention(nn.Module):
    """
    在每个窗口内部计算多头注意力

    参数：
        dim：Token通道数
        num_heads：注意力头数，dim必须能被其整除

    输入：
        windows：[窗口总数, 每窗Token数, dim]

    返回：
        output：注意力处理后的Token，形状与输入相同
        attention：注意力权重，
    """
    def __init__(self,dim,num_heads,window_size):
        super().__init__()

        self.window_size = window_size
        self.qkv = WindowQKV(dim=dim,num_heads=num_heads)

         # 缩放系数：1 / sqrt(每头通道数)
        self.scale = self.qkv.head_dim ** -0.5

        self.relative_position_bias = RelativePositionBias(
            window_size=window_size,
            num_heads=num_heads
        )

        # 合并多个头后，用线性层混合通道信息
        self.proj = nn.Linear(dim, dim)

    def forward(self, windows,mask=None):
        """
        计算窗口注意力，可选加入移位窗口掩码。

        参数：
            windows：[窗口总数, Token数, 通道数]
            mask：None，或[每张图窗口数, Token数, Token数]
                0表示允许注意，-inf表示禁止注意

        返回：
            output：与windows形状相同
            attention：[窗口总数, 头数, Token数, Token数]
        """
        if windows.ndim != 3:
            raise ValueError("输入必须为[窗口总数, Token数, 通道数]")


        num_windows, num_tokens, channels = windows.shape

        if num_tokens != self.window_size ** 2:
            raise ValueError("Token数量必须等于窗口边长的平方")

        #[窗口数,头数,Token数,每头通道数]
        q,k,v = self.qkv(windows) 

        # K最后两维交换：
        # [256,3,16,8] → [256,3,8,16]
        #
        # 矩阵乘法：
        # [16,8] @ [8,16] → [16,16]
        scores = (q * self.scale) @ k.transpose(-2, -1)

        # [头数,Token数,Token数]
        bias = self.relative_position_bias()

        scores = scores + bias.unsqueeze(0)

        if mask is not None:
            if mask.ndim != 3 or mask.shape[1:] != (
                num_tokens, num_tokens
            ):
                raise ValueError("掩码形状不正确")

            windows_per_image = mask.shape[0]

            if windows_per_image == 0:
                raise ValueError("每张图的窗口数必须大于0")

            if num_windows % windows_per_image != 0:
                raise ValueError("窗口总数与掩码窗口数不匹配")

            if mask.device != scores.device:
                raise ValueError("掩码与特征必须位于同一设备")
            num_heads = q.shape[1]

            # [总窗口数,头数,Token数,Token数]
            # → [样本数,每图窗口数,头数,Token数,Token数]
            scores = scores.reshape(
                -1,
                windows_per_image,
                num_heads,
                num_tokens,
                num_tokens
            )

             # 对每个样本、每个注意力头应用同一组空间掩码
            scores = scores + mask.to(
                dtype=scores.dtype
            ).unsqueeze(0).unsqueeze(2)

            scores = scores.reshape(
                num_windows, num_heads, num_tokens, num_tokens
            )


        # 对每个查询Token，将其对所有键Token的权重归一化
        attention = scores.softmax(dim=-1)

        # [16,16] @ [16,8] → [16,8]
        output = attention @ v

        # [窗口数,头数,Token数,每头通道数]
        # → [窗口数,Token数,头数,每头通道数]
        output = output.transpose(1, 2).contiguous()

        # 将3个头的8维结果合并为24维
        output = output.reshape(
            num_windows, num_tokens, channels
        )

        output = self.proj(output)

        return output, attention

# -----------------6.检查窗口注意力-----------------------
def check_window_attention():
    """
    检查窗口注意力的形状、权重归一化和梯度。

    参数：
        无

    返回：
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = WindowAttention(dim=24, num_heads=3,window_size=4)
    windows = torch.randn(
        256, 16, 24,
        requires_grad=True
    )

    output, attention = model(windows)

    assert output.shape == windows.shape
    assert attention.shape == (256, 3, 16, 16)
    assert torch.isfinite(output).all().item()
    assert torch.isfinite(attention).all().item()
    assert (attention >= 0).all().item()

    # 每个查询Token对所有键Token的权重之和应为1
    weight_sums = attention.sum(dim=-1)

    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        atol=1e-6,
        rtol=1e-5
    )

    # 验证输出能将梯度传回输入及所有模型参数
    loss = output.square().mean()
    loss.backward()

    # 确认位置偏置已经参与注意力计算，并收到梯度
    bias_grad = model.relative_position_bias.bias_table.grad

    assert bias_grad is not None
    assert torch.isfinite(bias_grad).all().item()
    assert bias_grad.abs().sum().item() > 0

    print("相对位置偏置已接入注意力，梯度检查通过")

    assert windows.grad is not None
    assert torch.isfinite(windows.grad).all().item()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("注意力权重形状：", attention.shape)
    print("窗口注意力输出：", output.shape)
    print("窗口注意力形状、归一化与梯度检查通过")

#3.主程序
def main():
    check_window_qkv()
    check_window_attention()
    check_relative_position_bias()

if __name__ == "__main__":
    main()



        