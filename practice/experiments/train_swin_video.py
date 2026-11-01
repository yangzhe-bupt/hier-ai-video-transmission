# ======================================================================
# 文件：train_swin_video.py
# 位置：practice/experiments
# 功能：训练Swin编码器和CNN解码器，保存最佳权重及逐轮指标。
# 类与函数汇总：
#   - save_swin_checkpoint：保存架构配置、权重及指标。
#   - load_swin_models：按保存配置创建并加载模型。
#   - save_history：将每轮训练与验证MSE保存为CSV。
#   - save_latest_training：保存最近完整轮的模型、优化器、指标和随机状态。
#   - restore_training：恢复最近完整轮，继续下一轮训练。
#   - main：最多训练500轮，支持早停、独立目录和轮间恢复。
# 复用函数：
#   - train_one_epoch、evaluate：来自train_video。
# 输入：[B,T,3,128,128]，RGB浮点张量，范围[0,1]。
# 特征：[B,T,24,32,32]。
# 运行：python D:/learn/pytorch/practice/run.py experiments/train_swin_video.py
# 当前状态：支持最佳权重评测与完整轮边界续训；中途停止会重做未完成轮。
# ======================================================================

from pathlib import Path
from datetime import datetime
import math
import torch
from torch.utils.data import DataLoader
from swin_encoder import SwinGOPEncoder
from semantic_decoder import GOPDecoder
from video_folder_dataset import VideoFolderDataset
from train_video import train_one_epoch, evaluate
# -----------------1.保存最佳模型-----------------------
def save_swin_checkpoint(
    encoder, decoder, epoch, train_mse, val_mse, save_path
):
    """
    保存Swin编码器、CNN解码器及架构配置。

    参数：
        encoder：当前Swin编码器
        decoder：当前CNN解码器
        epoch：当前完成的训练轮数
        train_mse：本轮平均训练MSE
        val_mse：本轮验证MSE
        save_path：检查点保存路径
    返回：
        无；保存.pt文件
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": "swin_encoder_cnn_decoder_v1",

        # 加载时按照这些配置重新创建模型
        "config": {
            "embed_dim": 24,
            "num_heads": 3,
            "window_size": 4
        },

        "epoch": epoch,
        "train_mse": train_mse,
        "val_mse": val_mse,

        # 转到CPU保存，方便在不同设备上加载
        "encoder": {
            name: value.detach().cpu().clone()
            for name, value in encoder.state_dict().items()
        },
        "decoder": {
            name: value.detach().cpu().clone()
            for name, value in decoder.state_dict().items()
        }
    }

    torch.save(checkpoint, save_path)

    print("最佳模型已保存：", save_path)

# -----------------2.加载Swin检查点-----------------------
def load_swin_models(checkpoint_path, device):
    """
    根据检查点配置创建模型，并加载权重。

    参数：
        checkpoint_path：Swin版本检查点路径
        device：加载后模型所在设备

    返回：
        encoder：已加载权重的Swin编码器
        decoder：已加载权重的CNN解码器
        checkpoint：包含配置、轮数和指标的检查点字典
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["model_version"] != "swin_encoder_cnn_decoder_v1":
        raise ValueError("检查点架构与当前加载函数不匹配")

    config = checkpoint["config"]

    encoder = SwinGOPEncoder(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        window_size=config["window_size"]
    ).to(device)

    decoder = GOPDecoder(
        feature_channels=config["embed_dim"]
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()
    return encoder, decoder, checkpoint

# -----------------3.保存逐轮指标-----------------------
def save_history(history, save_path):
    """
    将训练、验证指标写入CSV。

    参数：
        history：记录列表，每项包含epoch、train_mse、val_mse
        save_path：CSV保存路径

    返回：
        无
    """
    import csv
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_mse", "val_mse"]
        )

        writer.writeheader()
        writer.writerows(history)

# -----------------完整轮边界续训-----------------------
def save_latest_training(
    encoder, decoder, optimizer, generator, epoch, history,
    best_val_mse, early_stop_best, bad_epochs, save_path, settings
):
    """
    保存最近一轮完整训练状态，而非仅最佳评测权重。

    参数：
        encoder、decoder：当前两侧模型；optimizer：当前Adam优化器。
        generator：训练DataLoader使用的随机生成器。
        epoch：已完整完成训练与验证的轮数；history：逐轮指标。
        best_val_mse：最低验证误差；early_stop_best：早停参考误差。
        bad_epochs：连续未明显改善轮数；save_path：latest.pt路径。
        settings：训练配置记录，恢复时应保持数据及配置一致。
    返回：无。先写临时文件再替换，保留旧latest直至新保存完成。
    """
    state = {
        "model_version": "swin_encoder_cnn_decoder_v1",
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch, "history": history,
        "best_val_mse": best_val_mse,
        "early_stop_best": early_stop_best, "bad_epochs": bad_epochs,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng": generator.get_state(), "settings": settings
    }
    save_path = Path(save_path)
    temp_path = save_path.with_suffix(".tmp")
    torch.save(state, temp_path)
    temp_path.replace(save_path)


def restore_training(path, encoder, decoder, optimizer, generator, device):
    """
    恢复完整轮边界状态，接着训练下一轮。

    参数：path为latest.pt；其余为已创建的模型、优化器、数据生成器和设备。
    返回：state字典，包含已完成轮数、指标和早停计数。
    说明：相同数据、设备和配置下恢复随机状态；不保证跨设备逐位一致。
    """
    state = torch.load(path, map_location=device, weights_only=True)
    if state["model_version"] != "swin_encoder_cnn_decoder_v1":
        raise ValueError("续训检查点架构不匹配")
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    generator.set_state(state["loader_rng"].cpu())
    if device.type == "cuda" and state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    return state


# -----------------4.多轮训练入口-----------------------
def main():
    """
    最多训练500轮，逐轮验证、保存并支持早停；最后复查最佳模型。

    参数：
        无
    返回：
        无；输出最佳模型和逐轮指标文件
    """

    torch.manual_seed(24)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # 通过practice/run.py运行，沿用兼容的逻辑路径
    project_dir = Path(__file__).resolve().parent.parent
    videos_dir = project_dir / "data" / "videos"

    # ----------本次实验设置----------
    epochs = 500       # 总轮数上限，恢复训练时也表示累计轮数上限
    patience = 30       # 连续30轮未明显改善则提前停止
    min_delta = 1e-5    # 验证MSE下降超过此值才重置早停计数

    # 首次运行保持None；续训时填入控制台打印的实验目录。
    # 例如：r"D:\learn\pytorch\outputs\swin_runs\20261101_200000_123456"
    resume_run_dir = None

    if resume_run_dir is None:
        # 每次新实验使用独立目录，避免覆盖此前的5轮基线。
        run_dir = (project_dir / "outputs" / "swin_runs"
                   / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = Path(resume_run_dir)
        if not (run_dir / "latest.pt").is_file():
            raise FileNotFoundError("该实验目录中没有latest.pt")

    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    latest_path = run_dir / "latest.pt"
    print("实验目录：", run_dir)

    train_dataset = VideoFolderDataset(
        video_dir=videos_dir / "train",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    val_dataset = VideoFolderDataset(
        video_dir=videos_dir / "val",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    generator = torch.Generator().manual_seed(24)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 模型、优化器都在循环外创建：
    # 后续每轮继续更新同一组参数
    encoder = SwinGOPEncoder(
        embed_dim=24,
        num_heads=3,
        window_size=4
    ).to(device)

    decoder = GOPDecoder(feature_channels=24).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )
    epochs = 500
    best_val_mse = float("inf")
    history = []
    print("运行设备：", device)
    for epoch in range(0, epochs + 1):
        print(f"\n===== Swin第{epoch}/{epochs}轮 =====")
        train_mse = train_one_epoch(
            encoder, decoder, train_loader, optimizer, device
        )
        val_mse = evaluate(
            encoder, decoder, val_loader, device
        )
        print(
            f"训练MSE：{train_mse:.6f}，"
            f"验证MSE：{val_mse:.6f}"
        )

        if not math.isfinite(train_mse) or not math.isfinite(val_mse):
            raise RuntimeError("MSE出现NaN或无穷，请检查训练；上一完整轮保存在latest.pt")
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse
        })
        # 每轮保存指标，便于之后画曲线
        save_history(history, history_path)
        # 当前轮验证结果更好时，覆盖本实验的最佳检查点
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_swin_checkpoint(
                encoder=encoder,
                decoder=decoder,
                epoch=epoch,
                train_mse=train_mse,
                val_mse=val_mse,
                save_path=checkpoint_path
            )
        # 最佳权重保存任何下降；早停只响应超过min_delta的改善。
        if val_mse < early_stop_best - min_delta:
            early_stop_best = val_mse
            bad_epochs = 0
        else:
            bad_epochs += 1

        save_latest_training(
            encoder, decoder, optimizer, generator, epoch, history,
            best_val_mse, early_stop_best, bad_epochs, latest_path,
            {"seed": 24, "batch_size": 2, "gop_size": 4,
             "crop_size": 128, "stride": 4, "learning_rate": 0.001,
             "train_dir": str(videos_dir / "train"),
             "val_dir": str(videos_dir / "val"),
             "patience": patience, "min_delta": min_delta}
        )
        print(f"早停计数：{bad_epochs}/{patience}")
        if bad_epochs >= patience:
            print("验证指标长期未明显改善，提前结束训练")
            break

    # 新建模型并加载最佳权重，而不是直接使用最后一轮模型
    loaded_encoder, loaded_decoder, checkpoint = load_swin_models(
        checkpoint_path, device
    )

    loaded_val_mse = evaluate(
        loaded_encoder, loaded_decoder, val_loader, device
    )
    assert abs(loaded_val_mse - checkpoint["val_mse"]) < 1e-6, (
        "加载后的验证MSE与保存记录不一致"
    )
    print(f"\n最佳轮数：{checkpoint['epoch']}")
    print(f"最佳验证MSE：{checkpoint['val_mse']:.6f}")
    print(f"加载后验证MSE：{loaded_val_mse:.6f}")
    print("最佳模型加载与验证一致性检查通过")
    print("训练记录：", history_path)
    
if __name__ == "__main__":
    main()