# ======================================================================
# 文件：evaluate_biran_integration.py
# 位置：practice/checks
# 功能：使用真实视频和已有权重，对比原始链路与毕然模块倍数1、2的重建MSE及图像。
# 类与函数汇总：
#   - compare_reconstruction：使用相同输入，相同权重，比较三条链路
#   - show_comparison：每行显示一种结果，每列对应一帧
#   - main：加载既有权重并评测配置指定的真实视频GOP。
# 运行：python D:/learn/pytorch/practice/run.py checks/evaluate_biran_integration.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from check_checkpoint import load_models
from video_feature_demo import load_video_as_gop

# 导入毕然的模块
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "Biran"))

from feature_decomposition import PublicResidualDecomposer

#--------------------1.比较三条重建链路----------------
def compare_reconstruction(encoder,decoder,gop):
    """
    使用相同输入，相同权重，比较三条链路

    参数:
        encoder:已加载训练权重的编码器
        decoder:已加载训练权重的解码器
        gop:视频张量，[B,T,C,H,W]

    返回:
        results:字典，保存三组重建视频
    """
    encoder.eval()
    decoder.eval()

    results = {}

    with torch.no_grad():
        features = encoder(gop)

        #原有解编码链路
        baseline = decoder(features)
        results["A:Baseline"] = baseline.cpu()

        #加入毕然的分解和恢复模块
        for factor in (1,2):
            decomposer = PublicResidualDecomposer(downsample_factor = factor)

            parts = decomposer.encode(features)
            restored = decomposer.decode(parts)
            reconstructed = decoder(restored)

            assert restored.shape == features.shape
            assert reconstructed.shape == gop.shape
            assert torch.isfinite(reconstructed).all().item()

            # 倍数1应基本保持原结果
            if factor == 1:
                assert torch.allclose(
                    restored, features,
                    rtol=1e-5, atol=1e-6
                )
                assert torch.allclose(
                    reconstructed, baseline,
                    rtol=1e-5, atol=1e-6
                )

            label = f"{'B' if factor == 1 else 'C'}: Factor {factor}"
            results[label] = reconstructed.cpu()

    # MSE：所有样本、帧、通道和像素的平均平方误差
    target = gop.cpu()

    for name, reconstructed in results.items():
        mse = (reconstructed - target).square().mean().item()
        print(f"{name}，MSE={mse:.8f}")

    print("真实视频三条链路检查通过")

    return results

#---------------2.显示并保存对比图片---------------------
def show_comparison(gop,results,save_path):
    """
    每行显示一种结果，每列对应一帧

    参数：
        gop:原始视频张量
        results:三条链路的结果字典
        save_path:对比图片保存路径

    返回:
        无
        仅显示第一个样本
    """
    rows = {"Original": gop.detach().cpu(), **results}
    num_frames = gop.shape[1]

    fig, axes = plt.subplots(
        len(rows), num_frames,
        figsize=(3 * num_frames, 3 * len(rows)),
        squeeze=False
    )

    for row, (name, video) in enumerate(rows.items()):
        for t in range(num_frames):
            # [C,H,W] → [H,W,C]，方便Matplotlib显示
            image = video[0, t].permute(1, 2, 0).numpy()

            axes[row, t].imshow(image)
            axes[row, t].set_title(f"{name} | Frame {t}")
            axes[row, t].axis("off")

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    print("对比图片保存位置：", save_path)
    plt.show()
    plt.close(fig)

#主程序
def main():
    """
    加载已有模型，用视频开头4帧进行接口与重建对照。
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    video_path = project_dir / "data" / "毕洋洋.mp4"
    checkpoint_path = (
        project_dir / "outputs" / "single_gop_5000steps_v1.pt"
    )

    # 复用你已有的视频读取、中心裁剪和张量转换流程
    gop = load_video_as_gop(
        video_path=str(video_path),
        crop_size=128,
        num_frames=4,
        start_frame=100
    )
    gop = gop.to(device)

    # 加载训练好的参数，不进行训练和参数更新
    encoder, decoder = load_models(
        str(checkpoint_path), device
    )

    print("运行设备：", device)

    results = compare_reconstruction(encoder, decoder, gop)

    show_comparison(
        gop,
        results,
        project_dir / "outputs" / "biran_integration_frame100.png"
    )

#测试
if __name__=='__main__':
    main()

