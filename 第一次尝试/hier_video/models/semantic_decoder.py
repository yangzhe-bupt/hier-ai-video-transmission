import torch
import torch.nn as nn

class UpsampleBlock(nn.Module):
    """
    将特征空间尺寸扩大两倍，再用卷积处理

    参数：
        channels:输入和输出特征通道数

    输入：
        x = [N,channels,H,W]

    返回：
        features:[N，channels,2H,2W]
    """
    def __init__(self,channels):
        super().__init__()

        #高和宽扩大倍数，双线性插值，采用不对齐角点的坐标方式
        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
        #卷积：学习处理上采样后的特征
        self.conv = nn.Conv2d(channels,channels,3,1,1)
        #激活函数
        self.activation = nn.GELU()

    def forward(self,x):
        #放大尺寸->卷积->激活函数
        features = self.activation(self.conv(self.upsample(x)))

        return features


class GOPDecoder(nn.Module):
    """
    对GOP中的特征逐帧解码，恢复RGB图像尺寸

    参数：
        feature_channels:输入特征通道数

    输入：
        features:[B,T,feature_channels,Hz,Wz]

    返回：
        reconstructed:[B,T,3,4*Hz,4*Wz],数值范围[0,1]
    """
    def __init__(self,feature_channels):
        super().__init__()

        self.feature_channels = feature_channels

        #上采样
        self.upsample1 = UpsampleBlock(channels=feature_channels)
        self.upsample2 = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)

        #特征映射为RGB三通道
        self.to_rgb = nn.Conv2d(feature_channels,3,3,1,1)
        self.output_activation = nn.Sigmoid()

    def forward(self,features):
        if features.ndim != 5:
            raise ValueError("输入必须为[B,T,C,H,W]")

        B, T, C, H, W = features.shape

        if C != self.feature_channels:
            raise ValueError("输入通道数与解码器配置不一致")

        x = features.reshape(B*T,C,H,W)

        x = self.output_activation(self.to_rgb(self.upsample2(self.upsample1(x))))

        _, Cout, Hout, Wout = x.shape
        reconstructed = x.reshape(B, T, Cout, Hout, Wout)

        return reconstructed

