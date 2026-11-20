# ======================================================================
# 文件：check_biran_integration.py
# 位置：practice/checks
# 功能：使用模拟GOP检查编码器、毕然分解恢复模块和解码器的前向接口；覆盖CPU及可用CUDA、降采样倍数1和2。
# 类与函数汇总：
#   - check_biran_integration：检查：编码->特征分解->特征恢复->视频解码
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py checks/check_biran_integration.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import sys
from pathlib import Path
import torch
from semantic_encoder import GOPEncoder
from semantic_decoder import GOPDecoder

# 当前文件在 practice 文件夹中，上一层为项目目录
project_dir = Path(__file__).resolve().parent.parent

# 将毕然的文件夹加入模块搜索路径
sys.path.insert(0, str(project_dir / "Biran"))

from feature_decomposition import PublicResidualDecomposer

#----------------1.检查双方模块能否连接----------------
def check_biran_integration(device,downsample_factor):
    """
    检查：编码->特征分解->特征恢复->视频解码

    参数:
        device:运行设备
        downsample_factor:毕然模块的个体特征降采样倍数

    返回：
        无
        检查失败时抛出AssertionError
    """
    torch.manual_seed(24)

    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)

    decomposer = PublicResidualDecomposer(downsample_factor=downsample_factor)

    encoder.eval()
    decoder.eval()

    #构建模拟样本
    gop = torch.rand(2,4,3,128,128,device=device)

    with torch.no_grad():
        features = encoder(gop)

        #未经毕然模块
        reference = decoder(features)

        #毕然模块：分解公共特征，个体特征和元信息
        parts = decomposer.encode(features)
        #恢复
        restored_features = decomposer.decode(parts)

        reconstructed = decoder(restored_features)

         # 2. 检查形状
    assert features.shape == (2, 4, 24, 32, 32)

    assert parts.public.shape == (2, 1, 24, 32, 32)

    expected_size = 32 // downsample_factor
    assert parts.individual.shape == (
        2, 4, 24, expected_size, expected_size
    )

    assert restored_features.shape == features.shape
    assert reconstructed.shape == gop.shape

    # 3. 检查设备、类型和数值是否正常
    for name, tensor in [
        ("公共特征", parts.public),
        ("个体特征", parts.individual),
        ("恢复特征", restored_features),
        ("重建视频", reconstructed)
    ]:
        assert tensor.device == features.device, f"{name}设备不一致"
        assert tensor.dtype == features.dtype, f"{name}类型不一致"
        assert torch.isfinite(tensor).all().item(), f"{name}存在异常数值"

    assert reconstructed.min().item() >= 0
    assert reconstructed.max().item() <= 1

    # 不额外降采样时，分解后再恢复应与原特征基本一致
    if downsample_factor == 1:
        assert torch.allclose(
            restored_features, features,
            rtol=1e-5, atol=1e-6
        ), "分解与恢复后的特征不一致"

        assert torch.allclose(
            reconstructed, reference,
            rtol=1e-5, atol=1e-6
        ), "接入前后的解码结果不一致"

    print(f"\n设备：{device}，降采样倍数：{downsample_factor}")
    print("输入GOP：", gop.shape)
    print("编码特征：", features.shape)
    print("公共特征：", parts.public.shape)
    print("个体特征：", parts.individual.shape)
    print("恢复特征：", restored_features.shape)
    print("重建视频：", reconstructed.shape)
    print("双方模块前向对接检查通过")

#主程序
def main():
    """
    分别检查CPU，以及可用时的GPU。
    每个设备检查降采样倍数1和2。
    """
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        for factor in (1, 2):
            check_biran_integration(
                device=device,
                downsample_factor=factor
            )
#测试
if __name__=='__main__':
    main()

