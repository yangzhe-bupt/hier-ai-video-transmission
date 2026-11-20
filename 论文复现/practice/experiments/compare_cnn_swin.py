# ======================================================================
# 文件：compare_cnn_swin.py
# 位置：practice/experiments
# 功能：在相同验证GOP上比较CNN与Swin版本的重建结果。
# 类与函数汇总：
#   - compare_one_gop：评测一个GOP，保存三行重建对照图。
#   - main：加载两组模型，评测指定片段并保存CSV指标。
# 输入：[1,T,3,128,128]，RGB浮点张量，范围[0,1]。
# 图片排列：原图、CNN重建、Swin重建；每列对应同一源视频帧。
# 运行：python D:/learn/pytorch/practice/run.py experiments/compare_cnn_swin.py
# 当前状态：固定权重、无丢包重建对照，不进行训练。
# ======================================================================

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from check_checkpoint import load_models
from train_swin_video import load_swin_models
from video_feature_demo import load_video_as_gop


# -----------------1.对比一个GOP-----------------------
def compare_one_gop(
    cnn_encoder,
    cnn_decoder,
    swin_encoder,
    swin_decoder,
    gop,
    start_frame,
    output_dir
):
    """
    使用两组固定模型重建同一个GOP，并保存图片。

    参数：
        cnn_encoder、cnn_decoder：已加载最佳权重的CNN模型
        swin_encoder、swin_decoder：已加载最佳权重的Swin版本模型
        gop：视频张量，[1,T,3,H,W]，与模型处于同一设备
        start_frame：该GOP在源视频中的起始帧编号
        output_dir：图片保存目录

    返回：
        record：包含起始帧编号、两组MSE和图片路径的字典
    """
    # 固定为评测模式
    for model in (
        cnn_encoder, cnn_decoder,
        swin_encoder, swin_decoder
    ):
        model.eval()

    # 不建立计算图，也不更新模型参数
    with torch.no_grad():
        cnn_reconstructed = cnn_decoder(cnn_encoder(gop))
        swin_reconstructed = swin_decoder(swin_encoder(gop))

        assert cnn_reconstructed.shape == gop.shape
        assert swin_reconstructed.shape == gop.shape
        assert torch.isfinite(cnn_reconstructed).all().item()
        assert torch.isfinite(swin_reconstructed).all().item()

        # 在同一输入上统计整个GOP的平均像素MSE
        cnn_mse = (cnn_reconstructed - gop).square().mean().item()
        swin_mse = (swin_reconstructed - gop).square().mean().item()

    rows = [
        ("Original", gop.detach().cpu()),
        ("CNN", cnn_reconstructed.cpu()),
        ("Swin", swin_reconstructed.cpu())
    ]

    num_frames = gop.shape[1]

    fig, axes = plt.subplots(
        3,
        num_frames,
        figsize=(3 * num_frames, 9),
        squeeze=False
    )

    for row_index, (name, video) in enumerate(rows):
        for t in range(num_frames):
            # [C,H,W] → [H,W,C]，转换为绘图格式
            image = video[0, t].permute(1, 2, 0).numpy()

            axes[row_index, t].imshow(
                image,
                vmin=0,
                vmax=1
            )

            # 使用源视频中的实际帧编号
            axes[row_index, t].set_title(
                f"{name} | Frame {start_frame + t}"
            )
            axes[row_index, t].axis("off")

    fig.suptitle(
        f"CNN MSE={cnn_mse:.6f} | Swin MSE={swin_mse:.6f}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"compare_frame{start_frame}.png"
    fig.savefig(image_path, dpi=150, bbox_inches="tight")

    # 保存后关闭，避免多个图像占用内存
    plt.close(fig)

    print(
        f"帧{start_frame}～{start_frame + num_frames - 1}："
        f"CNN MSE={cnn_mse:.6f}，"
        f"Swin MSE={swin_mse:.6f}"
    )
    print("图片已保存：", image_path)

    return {
        "start_frame": start_frame,
        "cnn_mse": cnn_mse,
        "swin_mse": swin_mse,
        "image_path": str(image_path)
    }


# -----------------2.对比入口-----------------------
def main():
    """
    加载最佳模型，评测验证视频的三个固定GOP。

    参数：
        无

    返回：
        无；保存对照图片与CSV
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 经run.py运行时，使用兼容的原practice逻辑路径
    project_dir = Path(__file__).resolve().parent.parent
    outputs_dir = project_dir / "outputs"

    video_path = (
        project_dir / "data" / "videos" / "val"
        / "sintel_trailer_720p.mp4"
    )

    cnn_encoder, cnn_decoder = load_models(
        outputs_dir / "video_cnn_best_v1.pt",
        device
    )

    # 加载本次长训练选出的最佳模型，而不是最后一轮模型
    swin_checkpoint_path = (
        outputs_dir
        / "swin_runs"
        / "20261101_203819_892263"
        / "best.pt"
    )

    swin_encoder, swin_decoder, checkpoint = load_swin_models(
        swin_checkpoint_path,
        device
    )

    # 本次结果单独保存，不覆盖此前5轮Swin的对照图
    output_dir = outputs_dir / "cnn_swin_comparison_swin314"

    # 固定选择三个片段，不根据哪组模型表现好来挑选。
    # 初次查看后若片段是黑场或转场，需记录并另补有内容的片段。
    start_frames = [100, 400, 800]

    print("运行设备：", device)
    print("Swin最佳轮数：", checkpoint["epoch"])

    records = []

    for start_frame in start_frames:
        # 两组模型共享同一次读取与裁剪得到的输入
        gop = load_video_as_gop(
            video_path=str(video_path),
            crop_size=128,
            num_frames=4,
            start_frame=start_frame
        ).to(device)

        record = compare_one_gop(
            cnn_encoder=cnn_encoder,
            cnn_decoder=cnn_decoder,
            swin_encoder=swin_encoder,
            swin_decoder=swin_decoder,
            gop=gop,
            start_frame=start_frame,
            output_dir=output_dir
        )
        records.append(record)

    csv_path = output_dir / "comparison_metrics.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "start_frame",
                "cnn_mse",
                "swin_mse",
                "image_path"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    print("对照指标已保存：", csv_path)


if __name__ == "__main__":
    main()