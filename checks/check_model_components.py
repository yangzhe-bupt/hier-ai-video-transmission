import torch
from hier_video.models.semantic_encoder import *
from hier_video.models.semantic_decoder import *

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


if __name__ == '__main__':
    check_patch_embedding()
    check_downsample_block()
    check_gop_encoder()
    check_upsample_block()
    check_gop_decoder()
