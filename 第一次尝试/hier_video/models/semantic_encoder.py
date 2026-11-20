import torch
import torch.nn as nn

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

        if H % 4 != 0 or W % 4 != 0:
            raise ValueError("高、宽必须能被4整除")

        frames = gop.reshape(B*T,C,H,W)

        #卷积
        frame_features = self.patch_embed(frames)
        #降采样
        frame_features = self.downsample(frame_features)

        _,Cz,Hz,Wz = frame_features.shape

        features = frame_features.reshape(B,T,Cz,Hz,Wz)

        return features


def encode_gop_for_inference(model,gop):
    """
    使用已有模型提取视频GOP特征
    
    参数：
        model:已创建的GOPEncoder对象
        gop:RGB视频张量,[B,T,C,H,W],范围[0,1]
    
    返回:
        features:[B,T,embed_dim,H/4,W/4]
    """
    model.eval()

    with torch.no_grad():#不计算梯度少推理内存开销
        features = model(gop)

    print("视频GOP形状：", gop.shape)
    print("编码特征形状：", features.shape)
    print("特征数据类型：", features.dtype)

    return features

