# ======================================================================
# 文件：check_autoencoder.py
# 位置：practice/checks
# 功能：检查无丢失编解码链路的输入输出、梯度与参数更新。
# 类与函数汇总：
#   - check_autoencoder：检查无丢失编解码链路的形状和梯度
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py checks/check_autoencoder.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
import torch.nn as nn
from semantic_encoder import GOPEncoder
from semantic_decoder import GOPDecoder

#检查编解码链路
def check_autoencoder():
    """
    检查无丢失编解码链路的形状和梯度

    参数：
        无

    返回：
        无
        检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    #编码器，输出特征通道为24
    encoder = GOPEncoder(embed_dim=24)
    #解码器,输入特征通道为24
    decoder = GOPDecoder(feature_channels=24)

    optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=0.001)

    #保存更新前的参数
    encoder_before = [
        p.detach().clone() for p in encoder.parameters()
    ]
    decoder_before = [
        p.detach().clone() for p in decoder.parameters()
    ]

    optimizer.zero_grad()

    #测试模拟张量
    gop = torch.rand(1,4,3,128,128)

    features = encoder(gop)
    reconstructed = decoder(features)

    assert features.shape == (1, 4, 24, 32, 32)
    assert reconstructed.shape == gop.shape
    assert torch.isfinite(reconstructed).all().item()

    #计算重建像素与输入像素的平均均方误差
    criterion = nn.MSELoss()
    loss = criterion(reconstructed,gop)

    assert torch.isfinite(loss).item()

    loss.backward()

    for model_name, model in [
        ("encoder", encoder),
        ("decoder", decoder)
    ]:
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, (
                f"{model_name}.{name}没有梯度"
            )
            assert torch.isfinite(parameter.grad).all().item(), (
                f"{model_name}.{name}梯度存在异常"
            )

    optimizer.step()

    # 检查两端是否至少有一个参数发生变化
    encoder_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            encoder_before, encoder.parameters()
        )
    )

    decoder_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            decoder_before, decoder.parameters()
        )
    )

    assert encoder_changed, "编码器参数没有更新"
    assert decoder_changed, "解码器参数没有更新"

    print("编码器与解码器参数更新检查通过")

    print("编码特征形状：", features.shape)
    print("重建视频形状：", reconstructed.shape)
    print("初始MSE：", loss.item())
    print("编解码链路与梯度检查通过")

#主程序
def main():
    check_autoencoder()

#测试
if __name__=='__main__':
    main()