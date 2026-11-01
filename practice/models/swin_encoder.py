# ======================================================================
# 文件：swin_encoder.py
# 位置：practice/models
# 功能：结合Patch Embedding、Swin块对和降采样，提取视频GOP特征。
# 类与函数汇总：
#   - SwinGOPEncoder：使用共享参数逐帧编码GOP。
#   - check_swin_encoder：检查输出接口、解码器衔接及梯度。
#   - main：运行检查。
# 输入：[B,T,3,H,W]，RGB浮点张量，范围[0,1]。
# 输出：[B,T,embed_dim,H/4,W/4]。
# 运行：python D:/learn/pytorch/practice/run.py models/swin_encoder.py
# 当前状态：轻量二维Swin编码器，尚未加入时序建模、MoE或量化。
# ======================================================================

import torch
import torch.nn as nn

from semantic_encoder import PatchEmbedding, DownsampleBlock
from semantic_decoder import GOPDecoder
from swin_block_pair import SwinBlockPair

# -----------------1.Swin GOP编码器-----------------------
class SwinGOPEncoder(nn.Module):
    """
    对GOP中的所有帧使用同一套编码参数。

    参数：
        embed_dim：输出特征通道数
        num_heads：注意力头数，必须能整除embed_dim
        window_size：窗口边长，至少为2

    输入：
        gop：[B,T,3,H,W]
        H、W必须能被4和2*window_size整除，
        且Patch Embedding后的高宽必须大于window_size。

    返回：
        features：[B,T,embed_dim,H/4,W/4]
    """
    def __init__(
        self,
        embed_dim=24,
        num_heads=3,
        window_size=4
    ):
        super().__init__()

        self.window_size = window_size
        
        # 组件1：将RGB图像映射到特征空间
        # 输入：[N,C,H,W]
        # 输出：[N,embed_dim,H/2,W/2]
        # 该卷积的权重会在训练过程中学习
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

        # 组件2：使用窗口注意力处理空间特征
        # 先在固定窗口中交换信息，再通过移位改变窗口划分。
        self.swin = SwinBlockPair(
            dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size
        )

        # 组件3：进一步提取局部特征并降低空间分辨率
        # [N,embed_dim,H/2,W/2] → [N,embed_dim,H/4,W/4]
        self.downsample = DownsampleBlock(channels=embed_dim)

    def forward(self,gop):
        """
        执行一次GOP编码。

        参数：
            gop：五维RGB视频张量，[B,T,3,H,W]。

        返回：
            编码特征：[B,T,embed_dim,H/4,W/4]。
        """
        # ----------输入格式检查----------

        if gop.ndim != 5:
            raise ValueError("输入必须为[B,T,C,H,W]")

        B, T, C, H, W = gop.shape

        if B <= 0 or T <= 0:
            raise ValueError("批量大小和帧数必须大于0")

        if C != 3:
            raise ValueError("输入必须为RGB三通道")

        # 两次空间缩小，总共将高宽缩小4倍。
        # 要保证最终输出尺寸恰好为H/4、W/4，要求能被4整除。
        if H % 4 or W % 4:
            raise ValueError("高宽必须能被4整除")

        # 第一次缩小后，需要能够划分成完整窗口。
        # H/2能被window_size整除，等价于H能被2*window_size整除。
        if H % (2 * self.window_size) or W % (2 * self.window_size):
            raise ValueError(
                "Patch Embedding后的高宽必须能被窗口边长整除"
            )

        # 当前移位窗口实现不处理只有一行或一列窗口的特殊情况。
        if min(H // 2, W // 2) <= self.window_size:
            raise ValueError(
                "Patch Embedding后的高宽必须大于窗口边长"
            )

        # ----------合并样本维和时间维----------
        frames = gop.reshape(B*T,C,H,W)

        # ----------提取Patch特征----------
        # [B*T,3,128,128] → [B*T,24,64,64]
        x = self.patch_embed(frames)

        # ----------执行Swin空间处理----------
        # 内部完成划窗、注意力、MLP、移位和窗口还原。
        # 输入输出均为[B*T,24,64,64]。
        x = self.swin(x)

        # ----------进一步降采样----------
        # [B*T,24,64,64] → [B*T,24,32,32]
        x = self.downsample(x)

        # 读取实际输出尺寸
        _, Cz, Hz, Wz = x.shape

        # ----------恢复GOP维度----------
        # [B*T,Cz,Hz,Wz] → [B,T,Cz,Hz,Wz]
        features = x.reshape(B, T, Cz, Hz, Wz)

        return features

# -----------------2.检查编码器与解码器衔接-----------------------
def check_swin_encoder():
    """
    检查新编码器能否正常连接原解码器。

    检查内容：
        1. 编码特征形状与原接口一致。
        2. 解码后恢复输入视频尺寸，且数值正常。
        3. 编码器和解码器的参数均能收到有限梯度。
        4. 同一帧单独编码与随GOP一起编码的结果对应。

    参数：
        无。

    返回：
        无；检查失败时抛出AssertionError。

    注意：
        此处使用随机初始化模型和模拟输入。
        只检查流程，不代表训练后的重建质量。
    """
    # 固定随机种子，便于重复检查
    torch.manual_seed(24)

    # 有CUDA时使用GPU，否则使用CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder = SwinGOPEncoder(
        embed_dim=24,
        num_heads=3,
        window_size=4
    ).to(device)

    # 解码器仍接收24通道特征，不需要修改接口
    decoder = GOPDecoder(feature_channels=24).to(device)

    # 模拟一个GOP，包含4帧128×128的RGB图像
    # torch.rand生成[0,1)范围内的随机数
    gop = torch.rand(1, 4, 3, 128, 128, device=device)

    # ----------前向检查----------

    features = encoder(gop)
    reconstructed = decoder(features)

    assert features.shape == (1, 4, 24, 32, 32)
    assert reconstructed.shape == gop.shape

    # 检查是否出现NaN或正负无穷
    assert torch.isfinite(features).all().item()
    assert torch.isfinite(reconstructed).all().item()

    # 原解码器末尾使用Sigmoid，输出应在[0,1]范围
    assert reconstructed.min().item() >= 0
    assert reconstructed.max().item() <= 1

    # ----------反向传播检查----------

    # 计算重建MSE，再沿整条链路反向传播。
    # 本次没有optimizer.step()，因此不会更新模型参数。
    loss = (reconstructed - gop).square().mean()
    loss.backward()

    for model_name, model in [
        ("encoder", encoder),
        ("decoder", decoder)
    ]:
        for name, parameter in model.named_parameters():
            # grad不是None，说明该参数参与了反向传播
            assert parameter.grad is not None, (
                f"{model_name}.{name}没有梯度"
            )

            # 梯度还需要是有限数值
            assert torch.isfinite(parameter.grad).all().item()

    # ----------帧对应关系检查----------

    encoder.eval()

    # 这里只比较输出，不需要建立新的计算图
    with torch.no_grad():
        batch_features = encoder(gop)

        # 1:2保留时间维：
        # [1,4,3,128,128] → [1,1,3,128,128]
        single_features = encoder(gop[:, 1:2])

    # 批量和逐帧计算可能存在很小的浮点差异，因此使用allclose
    assert torch.allclose(
        batch_features[:, 1:2],
        single_features,
        rtol=1e-4,
        atol=1e-5
    ), "批量与单帧编码结果不对应"

    print("运行设备：", device)
    print("Swin编码特征：", features.shape)
    print("重建GOP：", reconstructed.shape)
    print("样本与帧对应检查通过")
    print("Swin编码器与现有解码器衔接及梯度检查通过")


# -----------------3.主程序-----------------------
def main():
    check_swin_encoder()


if __name__ == "__main__":
    main()

# ======================== 编解码流程框图 ========================
#
# 输入视频GOP（RGB，数值范围[0,1]）
# [B,T,3,128,128]
#          │
#          ▼
# 合并样本维B与时间维T
# [B*T,3,128,128]
#          │
#          ▼
# ┌───────────────────────────────────────────┐
# │ Patch Embedding                           │
# │ 2×2卷积，步长2：映射通道并缩小空间尺寸       │
# │ [B*T,3,128,128] → [B*T,24,64,64]          │
# └───────────────────────────────────────────┘
#          │
#          ▼
# ┌───────────────────────────────────────────┐
# │ SwinBlockPair：基础二维Swin块对             │
# │                                           │
# │  非移位窗口块                             │
# │  划分4×4窗口                              │
# │    → LayerNorm → 窗口注意力 → 残差相加     │
# │    → LayerNorm → MLP → 残差相加            │
# │    → 窗口还原                             │
# │                 │                         │
# │                 ▼                         │
# │  移位窗口块                               │
# │  向上、向左循环移动2个位置 → 划分4×4窗口    │
# │    → LayerNorm → 带掩码注意力 → 残差相加   │
# │    → LayerNorm → MLP → 残差相加            │
# │    → 窗口还原 → 向下、向右移回2个位置      │
# │                                           │
# │  两种注意力均包含可学习的相对位置偏置       │
# │  输入输出均为 [B*T,24,64,64]              │
# └───────────────────────────────────────────┘
#          │
#          ▼
# ┌───────────────────────────────────────────┐
# │ DownsampleBlock                           │
# │ 3×3卷积，步长2，填充1 → GELU              │
# │ [B*T,24,64,64] → [B*T,24,32,32]           │
# └───────────────────────────────────────────┘
#          │
#          ▼
# 恢复样本维B与时间维T
# [B,T,24,32,32]
#          │
#          ├──── SwinGOPEncoder的输出接口 ────
#          │
#          ▼
# 现有GOPDecoder
# 合并B与T → 两次上采样及卷积 → Sigmoid → 恢复B与T
#          │
#          ▼
# 重建视频GOP
# [B,T,3,128,128]，数值范围[0,1]
#
# 说明：
#   1. B：批量GOP数；T：每个GOP的帧数。
#   2. 各帧共享参数，当前没有跨帧信息交互。
#   3. 窗口掩码用于屏蔽循环移位边界，不是丢包缺失掩码。
#   4. 当前链路尚未接入量化、分包、丢包和缺失恢复。
# ==============================================================