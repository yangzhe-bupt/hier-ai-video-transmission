# ======================================================================
# 文件：semantic_encoder.py
# 位置：practice/models
# 功能：轻量CNN编码器：共享逐帧Patch Embedding与降采样；当前不包含Swin或时序融合。
# 类与函数汇总：
#   - PatchEmbedding：将单帧图像的非重叠2X2区域映射为特征
#     方法：__init__、forward
#   - DownsampleBlock：提取局部特征信息，并将空间尺寸缩小为原来的一半
#     方法：__init__、forward
#   - GOPEncoder：对GOP逐帧提取特征，并进行空间降采样
#     方法：__init__、forward
#   - encode_gop_for_inference：在评测模式和无梯度条件下提取GOP特征。
#   - check_patch_embedding：检查输出形状，以及同一模型重复调用时输出是否一致
#   - check_downsample_block：检查降采样尺寸和梯度传递
#   - check_gop_encoder：检查形状，以及批量处理与逐帧处理是否对应
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/semantic_encoder.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# 接口：[B,T,3,H,W] → [B,T,embed_dim,H/4,W/4]（高宽为4的倍数）。
# 说明：逐帧共享CNN，无时序融合；推理函数沿用同一编码器输出尺寸。
# ======================================================================

import torch
import torch.nn as nn


# ==================== 1. 基础组件 ====================

# Patch Embedding模块
class PatchEmbedding(nn.Module):
    """
    将单帧图像的非重叠2X2区域映射为特征

    参数：
        embed_dim:输出特征通道数
    
    输入：
        x:[N,3,H,W] H,W需要能被2整除

    输出：
        features:[N,embed_dim,H/2,W/2]
    """
    def __init__(self,embed_dim):
        super().__init__()

        self.conv = nn.Conv2d(3,embed_dim,2,2)

    #前向传播
    def forward(self,x):

        return self.conv(x)


# 降采样模块
class DownsampleBlock(nn.Module):
    """
    提取局部特征信息，并将空间尺寸缩小为原来的一半

    参数：
        channels:输入，输出的特征通道数
    
    输入：
        x:[N,channels,H,W], H, W为偶数
    
    返回：
        features:[N,channels,H/2,W/2]
    """
    def __init__(self,channels):
        super().__init__()

        #注意此处边界补零
        self.conv = nn.Conv2d(channels,channels,3,2,1)
        #GELU：非线性激活函数，不改变张量形状
        self.activation = nn.GELU()

    def forward(self,x):
        #卷积->降采样->非线性变换
        features = self.activation(self.conv(x))

        return features


# ==================== 2. 高层组件 ====================

# GOP特征提取模块
class GOPEncoder(nn.Module):
    """
    对GOP逐帧提取特征，并进行空间降采样

    参数：
        embed_dim:输出特征通道数

    输入：
        GOP:[B,T,3,H,W]

    返回：
        features:[B,T,embed_dim,H/4,W/4]    
    """
    def __init__(self,embed_dim):
        super().__init__()

        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.downsample = DownsampleBlock(channels=embed_dim)

    def forward(self,gop):
        if gop.ndim != 5:
            raise ValueError("输入必须为[B,T,C,H,W]")

        B,T,C,H,W = gop.shape

        #print(f"B={B}, T={T}, C={C}, H={H}, W={W}")

        if C != 3:
            raise ValueError("输入必须为RGB三通道")

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError("高、宽必须能被2整除")

        frames = gop.reshape(B*T,C,H,W)

        #卷积
        frame_features = self.patch_embed(frames)
        #降采样
        frame_features = self.downsample(frame_features)

        _,Cz,Hz,Wz = frame_features.shape

        features = frame_features.reshape(B,T,Cz,Hz,Wz)

        return features


# 提取真实视频GOP特征
def encode_gop_for_inference(model,gop):
    """
    使用已有模型提取视频GOP特征
    
    参数：
        model:已创建的GOPFeatureExtractor对象
        gop:RGB视频张量,[B,T,C,H,W],范围[0,1]
    
    返回:
        features:[B,T,embed_dim,H/2,W/2]
    """
    model.eval()

    with torch.no_grad():#不计算梯度少推理内存开销
        features = model(gop)

    print("视频GOP形状：", gop.shape)
    print("编码特征形状：", features.shape)
    print("特征数据类型：", features.dtype)

    return features


# ==================== 3. 检查模块 ====================

# 检查Patch Embedding模块
def check_patch_embedding():
    """
    检查输出形状，以及同一模型重复调用时输出是否一致
    """
    torch.manual_seed(24)

    model = PatchEmbedding(embed_dim=24)

    #模拟单帧
    x = torch.randn(1,3,128,128)

    output1 = model(x)
    output2 = model(x)

    assert output1.shape == (1, 24, 64, 64)
    assert torch.equal(output1, output2)

    print("输出形状：", output1.shape)
    print("重复调用检查通过")


# 检查降采样模块
def check_downsample_block():
    """
    检查降采样尺寸和梯度传递

    参数：
        无
    
    返回：
        无
        检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = DownsampleBlock(24)

    x = torch.randn(4,24,64,64,requires_grad=True)

    output = model(x)

    assert output.shape == (4, 24, 32, 32)
    assert torch.isfinite(output).all().item()

    loss = output.square().mean()
    #计算梯度，但不会更新模型参数
    loss.backward()

    assert x.grad is not None
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all().item()

    print("降采样输出：", output.shape)
    print("降采样与梯度检查通过")


# 检查GOP特征
def check_gop_encoder():
    """
    检查形状，以及批量处理与逐帧处理是否对应
    
    参数：
        无
    
    返回:
        无：
        检查失败时抛出AsseertionError
    """
    torch.manual_seed(24)

    model = GOPEncoder(24)

    gop = torch.randn(2,4,3,128,128)

    features = model(gop)

    assert features.shape == (2, 4, 24, 32, 32)

    for b in range(gop.shape[0]):
        for t in range(gop.shape[1]):

            # 单独取出一帧，并保留长度为1的batch维度
            frame = gop[b:b + 1, t]
            expected = model.patch_embed(frame)
            expected = model.downsample(expected)

            # allclose：允许浮点计算存在微小差异
            assert torch.allclose(
                features[b:b + 1, t],
                expected,
                rtol=1e-5,
                atol=1e-6
            ), f"样本{b}、帧{t}的特征不对应"

    print("GOP特征形状：", features.shape)
    print("样本与帧对应检查通过")   


# ==================== 4. 主函数 ====================

#主函数
def main():
    #check_patch_embedding()
    check_gop_encoder()
    #check_downsample_block()


#测试
if __name__=='__main__':
    main()