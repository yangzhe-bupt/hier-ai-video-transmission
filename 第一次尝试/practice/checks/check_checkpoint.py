# ======================================================================
# 文件：check_checkpoint.py
# 位置：practice/checks
# 功能：加载已保存的编解码器权重，并检查保存加载前后的重建一致性。
# 类与函数汇总：
#   - load_models：创建模型，并加载保存的参数
#   - check_checkpoint：使用同一输入，比较原模型和重新加载模型的输出
# 运行：python D:/learn/pytorch/practice/run.py checks/check_checkpoint.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
from semantic_decoder import GOPDecoder
from semantic_encoder import GOPEncoder

#加载训练后的模型
def load_models(checkpoint_path,device):
    """
    创建模型，并加载保存的参数

    参数：
        chenkpoint_path:.pt文件路径
        device:运行设备

    返回：
        encoder:加载参数的编码器
        decoder:加载参数的解码器
    """
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=True)

    channels = checkpoint["feature_channels"]

    encoder = GOPEncoder(channels).to(device)
    decoder = GOPDecoder(channels).to(device)

    #加载参数
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    return encoder,decoder

#检查保存后是否一致
def check_checkpoint(encoder, decoder, gop, checkpoint_path):
    """
    使用同一输入，比较原模型和重新加载模型的输出

    参数:
        encoder: 保存前的编码器
        decoder: 保存前的解码器
        gop: 相同的视频输入，[B,T,3,H,W]
        checkpoint_path: 已保存的检查点路径

    返回:
        无
        检查失败时抛出AssertionError
    """
    device = next(encoder.parameters()).device
    inputs = gop.to(device)

    encoder.eval()
    decoder.eval()

    # 新建模型并加载，不复用原模型对象
    loaded_encoder, loaded_decoder = load_models(
        checkpoint_path,
        device
    )

    with torch.no_grad():
        expected = decoder(encoder(inputs))
        actual = loaded_decoder(loaded_encoder(inputs))

    # allclose：允许浮点运算产生微小误差
    assert torch.allclose(
        expected, actual,
        rtol=1e-5,
        atol=1e-6
    ), "加载后的重建结果不一致"

    max_error = (expected - actual).abs().max().item()

    print("加载前后最大差值：", max_error)
    print("检查点加载与重建一致性检查通过")