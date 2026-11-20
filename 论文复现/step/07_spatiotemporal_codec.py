"""Step07：先理解三维窗口的划分与还原。"""

import torch
import torch.nn as nn

import importlib

# 复用真实视频读取与一致裁剪函数。
step05 = importlib.import_module("05_video_dataset")

# 如果文件顶部尚未导入Path，再添加这一行。
from pathlib import Path

# 复用 Step06 的空间 Patch Merging。
step06 = importlib.import_module("06_spatial_codec")
PatchMerging = step06.PatchMerging
PatchExpansion = step06.PatchExpansion

def window_partition_3d(features, window_size):
    """
    将 BTCHW 特征划分为三维窗口。

    参数：
        features：[B,T,C,H,W]。
        window_size：(时间窗口长度, 空间窗口高, 空间窗口宽)。

    返回：
        windows：[全部窗口数, 每个窗口的位置数, C]。

    本轮要求 T、H、W 均能被对应窗口尺寸整除。
    """
    if features.ndim != 5:
        raise ValueError("输入必须为 [B,T,C,H,W]")

    batch, time, channels, height, width = features.shape
    wt, wh, ww = window_size

    if min(wt, wh, ww) <= 0:
        raise ValueError("窗口尺寸必须为正")

    if time % wt or height % wh or width % ww:
        raise ValueError("T、H、W 必须能被对应窗口尺寸整除")

    # [B,T,C,H,W] → [B,T,H,W,C]
    x = features.permute(0, 1, 3, 4, 2)

    # 每个维度拆成“窗口数量”和“窗口内部位置”。
    # [B, 时间窗口数, wt, 高度窗口数, wh, 宽度窗口数, ww, C]
    x = x.reshape(
        batch,
        time // wt, wt,
        height // wh, wh,
        width // ww, ww,
        channels,
    )

    # 把所有窗口编号放在前面，把窗口内部位置放在后面。
    # [B, 时间窗口数, 高度窗口数, 宽度窗口数, wt, wh, ww, C]
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)

    # 合并窗口编号，并将窗口内部三维位置展平成 token 序列。
    windows = x.reshape(
        -1,
        wt * wh * ww,
        channels,
    )

    return windows

def window_reverse_3d(windows, window_size, original_shape):
    """
    将三维窗口还原为 BTCHW 特征。

    original_shape 是划分前的 [B,T,C,H,W]。
    本函数严格逆转 partition 中的排列顺序。
    """
    batch, time, channels, height, width = original_shape
    wt, wh, ww = window_size

    if min(wt, wh, ww) <= 0:
        raise ValueError("窗口尺寸必须为正")

    if time % wt or height % wh or width % ww:
        raise ValueError("原始尺寸必须能被窗口尺寸整除")

    expected_shape = (
        batch * (time // wt) * (height // wh) * (width // ww),
        wt * wh * ww,
        channels,
    )

    if tuple(windows.shape) != expected_shape:
        raise ValueError(
            f"窗口形状应为 {expected_shape}，"
            f"实际为 {tuple(windows.shape)}"
        )

    # 恢复窗口编号与窗口内部位置。
    x = windows.reshape(
        batch,
        time // wt,
        height // wh,
        width // ww,
        wt, wh, ww,
        channels,
    )

    # 将各维度的窗口编号与内部位置重新相邻排列。
    # [B, 时间窗口数, wt, 高度窗口数, wh, 宽度窗口数, ww, C]
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)

    # 合并各维度，恢复 [B,T,H,W,C]。
    x = x.reshape(batch, time, height, width, channels)

    # 恢复项目使用的 BTCHW。
    return x.permute(0, 1, 4, 2, 3).contiguous()

class WindowAttention3D(nn.Module):
    """
    对已经划分好的三维窗口计算多头自注意力。

    输入和输出：
        [num_windows, num_tokens, dim]

    本轮只实现基础注意力。
    三维位置已经被展开为 token 序列，位置关系将在后续补入。
    """

    def __init__(self, dim=96, num_heads=3, window_size=(4, 4, 4)):
        super().__init__()

        if num_heads <= 0 or dim <= 0 or dim % num_heads != 0:
            raise ValueError("dim 必须为正，且能被正整数 num_heads 整除")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 用于缩放 QK^T，避免内积随特征维度增大而过大。
        self.scale = self.head_dim ** -0.5

        self.window_size = tuple(window_size)
        wt, wh, ww = self.window_size

        if min(wt, wh, ww) <= 0:
            raise ValueError("窗口尺寸必须为正")

        # 每个方向分别有 2*窗口长度-1 种相对位移。
        num_relative_positions = (
            (2 * wt - 1)
            * (2 * wh - 1)
            * (2 * ww - 1)
        )

        # 可学习参数：每种相对位置、每个注意力头各有一个偏置。
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_relative_positions, num_heads)
        )

        # 构造窗口中每个位置的三维坐标。
        # 顺序为 t、h、w，与 window_partition_3d 展开窗口的顺序一致。
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(wt),
                torch.arange(wh),
                torch.arange(ww),
                indexing="ij",
            )
        )

        # [3,wt,wh,ww] → [3,L]，L为窗口内位置数。
        coords_flat = coords.flatten(1)

        # 计算任意两个位置之间的坐标差。
        # [3,L,1] - [3,1,L] → [3,L,L]
        relative_coords = (
            coords_flat[:, :, None] - coords_flat[:, None, :]
        )

        # [3,L,L] → [L,L,3]。
        # 最后一维依次是 Δt、Δh、Δw。
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # 将包含负数的位移转换为从0开始的索引。
        relative_coords[:, :, 0] += wt - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1

        # 将三维相对位置编码成一个整数索引。
        # index = t_index*(高度种类数*宽度种类数)
        #       + h_index*宽度种类数
        #       + w_index
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= 2 * ww - 1

        relative_position_index = relative_coords.sum(dim=-1)

        # 索引是固定数据，无须学习。
        # 注册为 buffer 后，会随模型移动到CPU/GPU，并保存进state_dict。
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

        # 用较小的随机值初始化可学习偏置。
        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

        # 一次线性映射同时生成 Q、K、V。
        # 输入96维，输出3×96维。
        self.qkv = nn.Linear(dim, 3 * dim)

        # 拼回所有注意力头后，再进行一次可学习的特征映射。
        self.proj = nn.Linear(dim, dim)

    def forward(self, windows, mask=None):
        if windows.ndim != 3 or windows.shape[-1] != self.dim:
            raise ValueError("输入必须为 [窗口数,位置数,dim]")
        
        num_windows, num_tokens, _ = windows.shape

        # [Wn,L,C] → [Wn,L,3C]
        qkv = self.qkv(windows)

        # 拆出 Q/K/V 和各个注意力头。
        # [Wn,L,3C] → [Wn,L,3,num_heads,head_dim]
        qkv = qkv.reshape(
            num_windows,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        # 把 Q/K/V 维移到最前面，方便分别取出。
        # [3,Wn,num_heads,L,head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 三者均为 [Wn,num_heads,L,head_dim]。
        q, k, v = qkv.unbind(dim=0)

        # 计算每个位置与窗口内所有位置的相关分数。
        # [Wn,num_heads,L,D] @ [Wn,num_heads,D,L]
        # → [Wn,num_heads,L,L]
        scores = (q * self.scale) @ k.transpose(-2, -1)

        expected_tokens = (
            self.window_size[0]
            * self.window_size[1]
            * self.window_size[2]
        )
        if num_tokens != expected_tokens:
            raise ValueError("输入位置数与窗口尺寸不匹配")

        # 为每一对位置查找对应的偏置。
        # 索引展开后有 L*L 项，查表结果为 [L*L,num_heads]。
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]

        # [L*L,num_heads] → [L,L,num_heads]
        relative_bias = relative_bias.reshape(
            num_tokens,
            num_tokens,
            self.num_heads,
        )

        # 调整为 [num_heads,L,L]，与注意力分数的最后三维对应。
        relative_bias = relative_bias.permute(2, 0, 1)

        # 增加窗口维，广播到所有窗口。
        # scores：[Wn,num_heads,L,L]
        # bias：  [1,num_heads,L,L]
        scores = scores + relative_bias.unsqueeze(0)

        # scores 此时形状为：
        # [B*每个样本的窗口数, num_heads, L, L]
        if mask is not None:
            # mask 为单个样本定义窗口边界规则，
            # 同一批次的其他样本使用相同规则。
            num_spatial_windows = mask.shape[0]

            if num_windows % num_spatial_windows != 0:
                raise ValueError("输入窗口数与掩码窗口数不匹配")

            if tuple(mask.shape[1:]) != (num_tokens, num_tokens):
                raise ValueError("掩码的位置数与输入窗口不匹配")

            batch = num_windows // num_spatial_windows

            # 拆出批次维，使掩码能广播到各样本、各注意力头。
            scores = scores.reshape(
                batch,
                num_spatial_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

            mask = mask.to(
                device=scores.device,
                dtype=scores.dtype,
            )

            # mask：[窗口数,L,L]
            # 扩展为：[1,窗口数,1,L,L]
            scores = scores + mask.unsqueeze(0).unsqueeze(2)

            # 恢复原布局，继续执行原有 softmax 和 AV。
            scores = scores.reshape(
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )

        # 每行权重非负且总和为1。
        attention = scores.softmax(dim=-1)

        # 使用注意力权重，对各位置的 V 加权求和。
        # [Wn,num_heads,L,L] @ [Wn,num_heads,L,D]
        # → [Wn,num_heads,L,D]
        output = attention @ v

        # 将各头移回同一个位置，再拼接通道。
        # [Wn,num_heads,L,D] → [Wn,L,num_heads,D]
        output = output.transpose(1, 2)

        # [Wn,L,num_heads,D] → [Wn,L,C]
        output = output.reshape(
            num_windows,
            num_tokens,
            self.dim,
        )

        return self.proj(output)

class SwinBlock3D(nn.Module):
    """
    支持非移位/空间移位、普通MLP或MoE前馈的3D Swin块。

    输入与输出：
        [B,T,C,H,W]，形状保持不变。

    结构：
        LayerNorm → 窗口注意力 → 残差相加
        LayerNorm → MLP → 残差相加
    """

    def __init__(
        self,
        dim=96,
        num_heads=3,
        window_size=(4, 4, 4),
        mlp_ratio=4,
        shift_size=(0, 0, 0),
    ):
        super().__init__()

        self.dim=dim
        self.window_size = tuple(window_size)

        self.shift_size = tuple(shift_size)

        # 本轮支持非移位，或时间不移位、两个空间方向都移位。
        # 掩码构造函数目前对应这两种使用方式。
        st, sh, sw = self.shift_size
        _, wh, ww = self.window_size

        if self.shift_size != (0, 0, 0):
            if st != 0 or not (0 < sh < wh and 0 < sw < ww):
                raise ValueError("当前只支持时间不移位的双向空间移位")

        # LayerNorm 要求待归一化的通道位于最后一维。
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attention = WindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
        )

        # MLP先扩展通道，再映射回原通道数。
        # dim=96、mlp_ratio=4时，隐藏层为384维。
        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, features, return_aux=False):
        if features.ndim != 5 or features.shape[2] != self.dim:
                raise ValueError("输入必须为 [B,T,dim,H,W]")
    
        _, time, _, height, width = features.shape
    
        # ---------- 第一条残差分支：注意力 ----------
        shortcut = features
    
        # 通道放到最后，执行LayerNorm。
        x = features.permute(0, 1, 3, 4, 2)
        x = self.norm1(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
    
        mask = None
        has_shift = self.shift_size != (0, 0, 0)
    
        if has_shift:
            # 先把特征向上、向左移动。
            x = torch.roll(
                x,
                shifts=tuple(-s for s in self.shift_size),
                dims=(1, 3, 4),
            )
    
            # 构造与当前尺寸对应的边界掩码。
            # 后续在attention中统一转换设备与类型。
            mask = build_spatial_shift_mask(
                time=time,
                height=height,
                width=width,
                window_size=self.window_size,
                shift_size=self.shift_size,
                )
    
        windows = window_partition_3d(x, self.window_size)
    
        # 非移位时mask=None；移位时屏蔽循环边界连接。
        windows = self.attention(windows, mask=mask)
    
        x = window_reverse_3d(
            windows,
            self.window_size,
            original_shape=features.shape,
        )
    
        if has_shift:
            # 逆移位，使输出重新对应原始位置。
            # 必须还原位置之后才能与shortcut相加。
            x = torch.roll(
                x,
                shifts=self.shift_size,
                dims=(1, 3, 4),
            )
    
        x = shortcut + x
    
        # ---------- 第二条残差分支：MLP ----------
        shortcut = x
    
        x = x.permute(0, 1, 3, 4, 2)
        x = self.norm2(x)
        # 前馈输入为[B,T,H,W,C]，最后一维是通道。
        # 默认只返回特征，兼容原来的nn.Sequential调用方式。
        if return_aux:
            if isinstance(self.mlp, nn.Sequential):
                # 原始MLP没有专家辅助损失，返回同设备的零标量。
                x = self.mlp(x)
                auxiliary_loss = x.new_zeros(())
            else:
                # 本项目替换进去的SparseMoE支持返回辅助损失。
                x, auxiliary_loss = self.mlp(x, return_aux=True)
        else:
            x = self.mlp(x)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
    
        output = shortcut + x

        # 辅助损失直接返回给调用者，保留其计算图。
        # 后续完整模型需要显式汇总各个MoE块的辅助损失。
        if return_aux:
            return output, auxiliary_loss
        return output

def check_window_attention():
    """检查划窗、注意力、还原以及反向传播的完整链路。"""
    features = torch.randn(
        1, 4, 96, 8, 8,
        requires_grad=True,
    )
    window_size = (4, 4, 4)

    windows = window_partition_3d(features, window_size)

    attention = WindowAttention3D(
        dim=96,
        num_heads=3,
    )
    attended_windows = attention(windows)

    print(
        "相对位置偏置表：",
        tuple(attention.relative_position_bias_table.shape),
    )
    print(
        "相对位置索引：",
        tuple(attention.relative_position_index.shape),
    )

    output = window_reverse_3d(
        attended_windows,
        window_size,
        original_shape=features.shape,
    )

    #print("注意力输入窗口：", tuple(windows.shape))
    #print("注意力输出窗口：", tuple(attended_windows.shape))
    #print("还原后的输出：", tuple(output.shape))

    assert attended_windows.shape == windows.shape
    assert output.shape == features.shape
    assert torch.isfinite(output).all()

    # 注意力会改变特征内容，这里检查形状和数值有限性。
    loss = output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    for name, parameter in attention.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

    print("窗口注意力梯度检查通过")

def check_swin_block_3d():
    """检查完整块的输出形状以及输入、参数的梯度。"""
    features = torch.randn(
        1, 4, 96, 8, 8,
        requires_grad=True,
    )

    block = SwinBlock3D(
        dim=96,
        num_heads=3,
        window_size=(4, 4, 4),
    )

    output = block(features)

    print("3D Swin块输入：", tuple(features.shape))
    print("3D Swin块输出：", tuple(output.shape))

    assert output.shape == features.shape
    assert torch.isfinite(output).all()

    loss = output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    # 包括归一化、QKV、位置偏置、输出映射和MLP。
    for name, parameter in block.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

    print("非移位3D Swin块梯度检查通过")

def check_spatial_shift():
    """用位置编号观察空间移位、窗口划分以及还原。"""

    # 给每个时空位置一个唯一编号，方便观察位置变化。
    # 这里只检查排列，因此用1个通道即可。
    features = torch.arange(
        4 * 8 * 8,
        dtype=torch.float32,
    ).reshape(1, 4, 1, 8, 8)

    shift_size = (0, 2, 2)
    window_size = (4, 4, 4)

    # 输入布局是 BTCHW。
    # 时间、高、宽分别对应第1、3、4维。
    #
    # 负数表示向前移动：
    # 高度方向向上2格，宽度方向向左2格。
    shifted = torch.roll(
        features,
        shifts=tuple(-s for s in shift_size),
        dims=(1, 3, 4),
    )

    # 移位后，仍按原来的规则划分窗口。
    windows = window_partition_3d(shifted, window_size)

    # 暂不计算注意力，先把窗口重新拼回特征图。
    merged = window_reverse_3d(
        windows,
        window_size,
        original_shape=features.shape,
    )

    # 正方向移动相同距离，恢复原来的位置。
    restored = torch.roll(
        merged,
        shifts=shift_size,
        dims=(1, 3, 4),
    )

    #print("移位前，第0帧：")
    #print(features[0, 0, 0])

    #print("移位后，第0帧：")
    #print(shifted[0, 0, 0])

    # 一个三维窗口按 t、h、w 顺序展开。
    # 前16个位置对应窗口内第一帧的4×4区域。
    #print("移位后，第一个窗口的第一帧：")
    #print(windows[0, :16, 0].reshape(4, 4))

    assert torch.equal(features, restored)
    print("移位、划窗、还原、逆移位检查通过")

def build_spatial_shift_mask(
    time,
    height,
    width,
    window_size=(4, 4, 4),
    shift_size=(0, 2, 2),
):
    """
    为“时间不移位、空间双向移位”的窗口构造注意力掩码。

    返回：
        [单个样本的窗口数, L, L]
        L = wt*wh*ww，为每个窗口的位置数。

    本轮要求时间窗口覆盖整个GOP，且高、宽都发生移位。
    """
    wt, wh, ww = window_size
    st, sh, sw = shift_size

    if wt != time or st != 0:
        raise ValueError("本轮要求时间窗口覆盖GOP，且时间不移位")

    if not (0 < sh < wh and 0 < sw < ww):
        raise ValueError("空间移位量必须大于0且小于窗口尺寸")

    if height % wh or width % ww:
        raise ValueError("高、宽必须能被窗口尺寸整除")

    # 用区域编号标记循环移位后需要彼此隔开的部分。
    # 时间维上的所有帧使用相同的空间区域编号。
    region_ids = torch.zeros(1, time, 1, height, width)

    # 每个空间方向划成三段。
    # 最后一个窗口范围内，再按移位量切出边界部分。
    h_slices = (
        slice(0, -wh),
        slice(-wh, -sh),
        slice(-sh, None),
    )
    w_slices = (
        slice(0, -ww),
        slice(-ww, -sw),
        slice(-sw, None),
    )

    region_id = 0

    for h_slice in h_slices:
        for w_slice in w_slices:
            region_ids[:, :, :, h_slice, w_slice] = region_id
            region_id += 1

    # 这些编号直接描述移位后窗口中的边界分组，
    # 因此这里不再对 region_ids 执行 roll。
    windows = window_partition_3d(region_ids, window_size)

    # 原来只有一个通道，将其移除。
    # [窗口数,L,1] → [窗口数,L]
    labels = windows.squeeze(-1)

    # 比较窗口内任意两个位置的区域编号。
    # [窗口数,L,1] 与 [窗口数,1,L] 广播，
    # 得到 [窗口数,L,L]。
    different_region = (
        labels.unsqueeze(2) != labels.unsqueeze(1)
    )

    # 同一区域允许连接，分数加0。
    # 不同区域禁止连接，分数加负无穷。
    mask = torch.zeros_like(different_region, dtype=torch.float32)
    mask = mask.masked_fill(different_region, float("-inf"))

    return mask

def check_shift_mask():
    mask = build_spatial_shift_mask(
        time=4,
        height=8,
        width=8,
        window_size=(4, 4, 4),
        shift_size=(0, 2, 2),
    )

    print("移位注意力掩码：", tuple(mask.shape))

    assert mask.shape == (4, 64, 64)

    # 第一个窗口没有循环跨边界，内部连接全部保留。
    assert (mask[0] == 0).all()

    # 每个位置始终允许关注自身。
    diagonal = mask.diagonal(dim1=-2, dim2=-1)
    assert (diagonal == 0).all()

    # 最后一个窗口，第一帧的第一行对应原图列：
    # 6、7、0、1。
    # 列6和列7允许连接，列6和列0禁止连接。
    assert mask[-1, 0, 1].item() == 0
    assert torch.isneginf(mask[-1, 0, 2]).item()

    #print("最后一个窗口，前4个位置之间的掩码：")
    #print(mask[-1, :4, :4])

    print("移位窗口掩码检查通过")

def check_shifted_swin_block():
        features = torch.randn(
        2, 4, 96, 8, 8,
        requires_grad=True,
        )

        block = SwinBlock3D(
            dim=96,
            num_heads=3,
            window_size=(4, 4, 4),
            shift_size=(0, 2, 2),
        )

        output = block(features)

        #print("移位块输入：", tuple(features.shape))
        #print("移位块输出：", tuple(output.shape))

        assert output.shape == features.shape
        assert torch.isfinite(output).all()

        loss = output.square().mean()
        loss.backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

        for name, parameter in block.named_parameters():
            if parameter.grad is None:
                raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

        print("移位3D Swin块梯度检查通过")

class SpatiotemporalStage(nn.Module):
    """
    一个时空编码阶段。

    结构：
        逐帧空间降采样
        → 非移位3D Swin块
        → 移位3D Swin块

    输入：
        [B,T,in_channels,H,W]

    输出：
        [B,T,out_channels,H/2,W/2]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=3,
        window_size=(4, 4, 4),
    ):
        super().__init__()
        
        self.in_channels = in_channels

        # 复用二维Patch Merging。
        # 临时合并B、T后，它会分别处理每一帧。
        self.merge = PatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        # 一对块交替使用固定窗口与空间移位窗口。
        self.blocks = nn.Sequential(
            SwinBlock3D(
                dim=out_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0),
            ),
            SwinBlock3D(
                dim=out_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(
                    0,
                    window_size[1] // 2,
                    window_size[2] // 2,
                ),
            ),
        )

    def forward(self, features, return_aux=False):
        if features.ndim != 5:
            raise ValueError("输入必须为 [B,T,C,H,W]")

        batch, time, channels, height, width = features.shape

        if channels != self.in_channels:
            raise ValueError("输入通道数与阶段配置不一致")

        # ---------- 逐帧空间降采样 ----------

        # [B,T,C,H,W] → [B*T,C,H,W]
        frames = features.reshape(
            batch * time,
            channels,
            height,
            width,
        )

        frames = self.merge(frames)

        # 从实际结果读取新的通道数与空间尺寸。
        _, new_channels, new_height, new_width = frames.shape

        # 恢复时间维，准备进行3D注意力计算。
        x = frames.reshape(
            batch,
            time,
            new_channels,
            new_height,
            new_width,
        )

        # ---------- 时空信息交互 ----------

        # 这里始终保留独立的时间维，
        # 使不同帧的位置能够进入同一个三维注意力窗口。
        if not return_aux:
            # 默认接口保持为单个特征张量，兼容原有模型和检查。
            return self.blocks(x)

        auxiliary_losses = []
        # 显式逐块执行，取出损失；nn.Sequential默认不会汇总损失。
        for block in self.blocks:
            x, auxiliary_loss = block(x, return_aux=True)
            auxiliary_losses.append(auxiliary_loss)

        # 阶段返回块损失之和，交给上层按实际块数统一取平均。
        # torch.stack保留计算图，不能在这里使用.item()或detach()。
        auxiliary_sum = torch.stack(auxiliary_losses).sum()
        return x, auxiliary_sum

class SpatiotemporalEncoder(nn.Module):
    """
    时空编码器接口验证版本。

    输入：[B,4,72,32,32]
    输出：[B,4,120,8,8]
    """

    def __init__(self):
        super().__init__()

        self.stage4 = SpatiotemporalStage(
            in_channels=72,
            out_channels=96,
        )

        self.stage5 = SpatiotemporalStage(
            in_channels=96,
            out_channels=120,
        )

    def forward(self, features, return_aux=False):
        if not return_aux:
            x = self.stage4(features)
            x = self.stage5(x)
            return x

        x, stage4_aux = self.stage4(features, return_aux=True)
        x, stage5_aux = self.stage5(x, return_aux=True)

        # 当前每阶段两个块，共四个。根据实际模块数量计算，避免写死。
        num_blocks = len(self.stage4.blocks) + len(self.stage5.blocks)
        auxiliary_mean = (stage4_aux + stage5_aux) / num_blocks
        return x, auxiliary_mean

class SpatiotemporalDecoderStage(nn.Module):
    """
    一个时空解码阶段。

    输入：
        [B,T,in_channels,H,W]

    输出：
        [B,T,out_channels,2H,2W]

    先使用3D Swin处理时空信息，再逐帧扩大空间尺寸。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=3,
        window_size=(4, 4, 4),
    ):
        super().__init__()

        self.in_channels = in_channels

        # 上采样之前，先在当前分辨率上处理特征。
        self.blocks = nn.Sequential(
            SwinBlock3D(
                dim=in_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0),
            ),
            SwinBlock3D(
                dim=in_channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(
                    0,
                    window_size[1] // 2,
                    window_size[2] // 2,
                ),
            ),
        )

        self.expansion = PatchExpansion(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(self, features, return_aux=False):
        if features.ndim != 5:
            raise ValueError("输入必须为 [B,T,C,H,W]")

        batch, time, channels, height, width = features.shape

        if channels != self.in_channels:
            raise ValueError("输入通道数与解码阶段配置不一致")

        # 3D注意力需要保留时间维。
        if return_aux:
            x = features
            auxiliary_losses = []
            # 显式取出每个块的辅助损失，随后仍正常执行上采样。
            for block in self.blocks:
                x, auxiliary_loss = block(x, return_aux=True)
                auxiliary_losses.append(auxiliary_loss)
            auxiliary_sum = torch.stack(auxiliary_losses).sum()
        else:
            x = self.blocks(features)

        # PatchExpansion接收NCHW，将B和T暂时合并。
        x = x.reshape(
            batch * time,
            channels,
            height,
            width,
        )

        # 逐帧空间上采样，同时调整通道数。
        x = self.expansion(x)

        _, new_channels, new_height, new_width = x.shape

        # 恢复视频布局，供后续时空模块处理。
        output = x.reshape(
            batch,
            time,
            new_channels,
            new_height,
            new_width,
        )
        if return_aux:
            return output, auxiliary_sum
        return output

class SpatiotemporalDecoder(nn.Module):
    """
    将时空特征恢复到空间编码器输出的尺寸。

    输入：[B,4,120,8,8]
    输出：[B,4,72,32,32]

    输出仍是特征，后面还需要空间解码器生成RGB。
    """

    def __init__(self):
        super().__init__()

        self.decode_stage5 = SpatiotemporalDecoderStage(
            in_channels=120,
            out_channels=96,
        )

        self.decode_stage4 = SpatiotemporalDecoderStage(
            in_channels=96,
            out_channels=72,
        )

    def forward(self, features, return_aux=False):
        if not return_aux:
            x = self.decode_stage5(features)
            x = self.decode_stage4(x)
            return x

        x, stage5_aux = self.decode_stage5(features, return_aux=True)
        x, stage4_aux = self.decode_stage4(x, return_aux=True)

        # 与编码器一致：先对块损失求和，再按实际块数取平均。
        num_blocks = (
            len(self.decode_stage5.blocks) + len(self.decode_stage4.blocks)
        )
        auxiliary_mean = (stage5_aux + stage4_aux) / num_blocks
        return x, auxiliary_mean

def check_spatiotemporal_encoder():
    # 模拟空间编码器的输出。
    # 这次先用随机特征，单独检查时空编码器。
    features = torch.randn(
        1, 4, 72, 32, 32,
        requires_grad=True,
    )

    encoder = SpatiotemporalEncoder()

    # 分别执行两阶段，方便观察中间尺寸。
    stage4_output = encoder.stage4(features)
    stage5_output = encoder.stage5(stage4_output)

    #print("时空编码输入：", tuple(features.shape))
    #print("Stage4 输出：", tuple(stage4_output.shape))
    #print("Stage5 输出：", tuple(stage5_output.shape))

    assert stage4_output.shape == (1, 4, 96, 16, 16)
    assert stage5_output.shape == (1, 4, 120, 8, 8)
    assert torch.isfinite(stage5_output).all()

    loss = stage5_output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    for name, parameter in encoder.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

    print("两阶段时空编码器梯度检查通过")

def check_spatiotemporal_decoder():
    features = torch.randn(
        1, 4, 120, 8, 8,
        requires_grad=True,
    )

    decoder = SpatiotemporalDecoder()

    middle = decoder.decode_stage5(features)
    output = decoder.decode_stage4(middle)

    #print("时空解码输入：", tuple(features.shape))
    #print("解码中间结果：", tuple(middle.shape))
    #print("时空解码输出：", tuple(output.shape))

    assert middle.shape == (1, 4, 96, 16, 16)
    assert output.shape == (1, 4, 72, 32, 32)
    assert torch.isfinite(output).all()

    loss = output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    for name, parameter in decoder.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"{name} 没有梯度")

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{name} 梯度存在异常")

    print("两阶段时空解码器梯度检查通过")

def check_full_video_path():
    """
    检查真实视频经过空间、时空编解码器的完整路径。

    所有模型均为新创建的随机权重模型。
    本轮检查形状、布局和数值是否正常。
    """
    torch.set_num_threads(2)

    step05 = importlib.import_module("05_video_dataset")

    video_path = (
        step06.PROJECT_ROOT
        / ".." / "pytorch" / "data" / "videos"
        / "train" / "big_buck_bunny_trailer_720p.mov"
    ).resolve()

    frames = step05.read_frames(
        video_path,
        start_frame=100,
        num_frames=4,
    )
    gop, _ = step05.frames_to_gop(
        frames,
        crop_size=256,
        seed=42,
    )

    spatial_encoder = step06.SpatialEncoder()
    temporal_encoder = SpatiotemporalEncoder()
    temporal_decoder = SpatiotemporalDecoder()
    spatial_decoder = step06.SpatialDecoder()

    # 所有模块切换为评估模式。
    for model in (
        spatial_encoder,
        temporal_encoder,
        temporal_decoder,
        spatial_decoder,
    ):
        model.eval()

    batch, time, channels, height, width = gop.shape

    with torch.no_grad():
        # ---------- 1. 空间编码 ----------
        # 二维编码器分别处理每帧，因此先合并B、T。
        frame_batch = gop.reshape(
            batch * time, channels, height, width
        )
        spatial_flat = spatial_encoder(frame_batch)

        # 恢复时间维，供3D时空编码器使用。
        _, cs, hs, ws = spatial_flat.shape
        spatial_features = spatial_flat.reshape(
            batch, time, cs, hs, ws
        )

        # ---------- 2. 时空编码 ----------
        temporal_features = temporal_encoder(spatial_features)

        # ---------- 3. 时空解码 ----------
        restored_spatial = temporal_decoder(temporal_features)

        # ---------- 4. 空间解码 ----------
        # 时空解码器输出BTCHW，再合并成二维解码器所需的NCHW。
        _, _, cr, hr, wr = restored_spatial.shape
        restored_flat = restored_spatial.reshape(
            batch * time, cr, hr, wr
        )
        reconstructed_flat = spatial_decoder(restored_flat)

        # 最后恢复视频的批次与时间维。
        reconstructed_gop = reconstructed_flat.reshape(
            batch, time, 3, height, width
        )

    #print("完整路径输入：", tuple(gop.shape))
    #print("空间编码结果：", tuple(spatial_features.shape))
    #print("时空编码结果：", tuple(temporal_features.shape))
    #print("时空解码结果：", tuple(restored_spatial.shape))
    #print("完整路径输出：", tuple(reconstructed_gop.shape))

    assert spatial_features.shape == (1, 4, 72, 32, 32)
    assert temporal_features.shape == (1, 4, 120, 8, 8)
    assert restored_spatial.shape == spatial_features.shape
    assert reconstructed_gop.shape == gop.shape

    for tensor in (
        spatial_features,
        temporal_features,
        restored_spatial,
        reconstructed_gop,
    ):
        assert torch.isfinite(tensor).all()

    assert reconstructed_gop.min() >= 0
    assert reconstructed_gop.max() <= 1

    print("真实视频完整编解码路径检查通过")

def main():
    torch.manual_seed(42)

    features = torch.randn(
        1,4,96,8,8,
        requires_grad=True,
    )

    window_size = (4,4,4)

    windows = window_partition_3d(features,window_size)

    restored = window_reverse_3d(
        windows,
        window_size,
        original_shape=features.shape,
    )

    #print("原始特征：", tuple(features.shape))
    #print("窗口序列：", tuple(windows.shape))
    #print("还原特征：", tuple(restored.shape))

    # 划分与还原只重新排列数据，数值应完全一致。
    assert torch.equal(features, restored)

    loss = restored.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    print("窗口划分与还原完全一致")
    print("梯度检查通过")

    check_window_attention()
    check_swin_block_3d()
    check_spatial_shift()
    check_shift_mask()
    check_shifted_swin_block()
    check_spatiotemporal_encoder()
    check_spatiotemporal_decoder()
    check_full_video_path()

if __name__ == "__main__":
    main()

