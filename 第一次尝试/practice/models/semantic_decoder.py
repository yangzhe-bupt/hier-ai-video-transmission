# ======================================================================
# 文件：semantic_decoder.py
# 位置：practice/models
# 功能：轻量CNN解码器：逐帧将特征恢复为RGB视频；当前不包含Swin或时序融合。
# 类与函数汇总：
#   - UpsampleBlock：将特征空间尺寸扩大两倍，再用卷积处理
#     方法：__init__、forward
#   - GOPDecoder：对GOP中的特征逐帧解码，恢复RGB图像尺寸
#     方法：__init__、forward
#   - check_upsample_block：检查输出形状及梯度传递
#   - check_gop_decoder：检查输出形状、数值范围及梯度传递
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py models/semantic_decoder.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# 接口：[B,T,C,Hz,Wz] → [B,T,3,4*Hz,4*Wz]，范围[0,1]。
# ======================================================================

import torch
import torch.nn as nn


# ==================== 1. 基础组件 ====================

# 上采样模块
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
        #卷积：起平滑作用
        self.conv = nn.Conv2d(channels,channels,3,1,1)
        #激活函数
        self.activation = nn.GELU()

    def forward(self,x):
        #放大尺寸->卷积->激活函数
        features = self.activation(self.conv(self.upsample(x)))

        return features


# ==================== 2. 高层组件 ====================

# GOP解码器
class GOPDecoder(nn.Module):
    """
    对GOP中的特征逐帧解码，恢复RGB图像尺寸

    参数：
        feature_channels:输入特征通道数

    输入：
        features:[B,T,feature_channels,Hz,Wz]

    返回：
        reconstructed:[B,T,C,4*Hz,4*Hz],数值范围[0,1]
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


# ==================== 3. 检查模块 ====================

# 检查上采样模块
def check_upsample_block():
    """
    检查输出形状及梯度传递

    参数：
        无
    
    返回：
        无
        检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = UpsampleBlock(channels=24)

    x = torch.randn(4,24,32,32,requires_grad=True)

    output = model(x)

    assert output.shape == (4, 24, 64, 64)
    assert torch.isfinite(output).all().item()

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert model.conv.weight.grad is not None
    assert torch.isfinite(x.grad).all().item()
    assert torch.isfinite(model.conv.weight.grad).all().item()

    print("上采样输出：", output.shape)
    print("上采样与梯度检查通过")


# 检查GOP解码器
def check_gop_decoder():
    """
    检查输出形状、数值范围及梯度传递

    参数:
        无

    返回:
        无；检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    model = GOPDecoder(feature_channels=24)

    features = torch.randn(
        2, 4, 24, 32, 32,
        requires_grad=True
    )

    reconstructed = model(features)

    assert reconstructed.shape == (2, 4, 3, 128, 128)
    assert torch.isfinite(reconstructed).all().item()
    assert reconstructed.min().item() >= 0
    assert reconstructed.max().item() <= 1

    # 仅检查梯度链路，不是正式训练损失
    loss = reconstructed.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all().item()

    # named_parameters()：依次返回参数名称和参数张量
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}没有梯度"
        assert torch.isfinite(parameter.grad).all().item()

    print("重建GOP形状：", reconstructed.shape)
    print("GOP解码器与梯度检查通过")


# ==================== 4. 主程序 ====================

#主程序
def main():
    #check_upsample_block()
    check_gop_decoder()


#测试
if __name__=='__main__':
    main()