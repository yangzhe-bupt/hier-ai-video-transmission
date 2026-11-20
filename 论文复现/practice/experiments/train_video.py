# ======================================================================
# 文件：train_video.py
# 位置：practice/experiments
# 功能：轻量CNN视频编解码训练：支持单轮和多轮训练、独立验证、最佳权重保存及加载指标复查。
# 类与函数汇总：
#   - train_one_epoch：遍历一次训练集，更新解码器和编码器参数
#   - evaluate：使用当前模型评测整个验证集，不更新参数
#   - save_training_checkpoint：保存当前模型权重和本轮测评指标
#   - check_saved_model：加载已保存的模型，重新评测验证集
#   - train_multiple_epochs：从头训练多轮，每轮验证，并保存最佳验证权重。
#   - train_main：组织本文件当前流程。
#   - evaluate_main：组织本文件当前流程。
# 运行：python D:/learn/pytorch/practice/run.py experiments/train_video.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from check_checkpoint import load_models
from semantic_encoder import GOPEncoder
from semantic_decoder import GOPDecoder
from video_folder_dataset import VideoFolderDataset


#-----------------1.训练一个epoch-----------------------
def train_one_epoch(encoder,decoder,loader,optimizer,device):
    """
    遍历一次训练集，更新解码器和编码器参数

    参数：
        encoder：编码器模型
        decoder：解码器模型
        loader：训练集DataLoader
        optimizer：管理编码器和解码器参数的优化器
        device：运行设备

    返回:
        epoch_mse:本轮样本数量加权平均MSE
    """
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_samples = 0

    for step,batch in enumerate(loader,start=1):
        #输入[B,T,C,H,W]
        gops = batch["gop"].to(device)

        optimizer.zero_grad(set_to_none = True)

        reconstructed = decoder(encoder(gops))

        loss = F.mse_loss(reconstructed,gops)

        loss.backward()
        optimizer.step()

        # 最后一批可能不足batch_size，因此按实际样本数统计
        batch_size = gops.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"批次 {step}/{len(loader)}，"
                f"当前批次MSE={loss.item():.6f}"
            )

    if total_samples == 0:
        raise ValueError("训练集没有可用样本")

    epoch_mse = total_loss / total_samples

    return epoch_mse

#------------------2.测试集测评------------------------
def evaluate(encoder, decoder, loader, device):
    """
    使用当前模型评测整个验证集，不更新参数

    参数：
        encoder：编码器模型
        decoder：解码器模型
        loader：验证集DataLoader
        device：运行设备

    返回：
        val_mse：按样本数量加权的平均验证MSE
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            gops = batch["gop"].to(device)

            reconstructed = decoder(encoder(gops))
            loss = F.mse_loss(reconstructed, gops)

            batch_size = gops.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("验证集没有可用样本")

    return total_loss / total_samples

#-----------------3.保存训练检查点--------------------
def save_training_checkpoint(encoder,decoder,device,epoch,train_mse,val_mse,save_path):
    """
    保存当前模型权重和本轮测评指标

    参数：
        encoder：训练后的编码器
        decoder：训练后的解码器
        device：当前运行设备，用于记录
        epoch：已完成的训练轮数
        train_mse：本轮平均训练MSE
        val_mse：本轮验证MSE
        save_path：检查点保存路径

    返回：
        无
        保存.pt文件并打印路径    
    """  
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": "lightweight_cnn_v1",
        "feature_channels": 24,
        "layout": "BTCHW",
        "training_scope": "video_folder",
        "epoch": epoch,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "device": str(device),

        # 转到CPU保存，方便之后在不同设备上加载
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
    print("检查点已保存：", save_path)

#-----------------4.加载检查点并验证--------------------
def check_saved_model():
    """
    加载已保存的模型，重新评测验证集

    参数：
        无

    返回：
        无
        检查加载后的MSE与保存时是否一致
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_dir = Path(__file__).resolve().parent.parent
    checkpoint_path = (
        project_dir / "outputs" / "video_cnn_epoch1_v1.pt"
    )

    #创建模型并加载权重
    encoder,decoder = load_models(
        str(checkpoint_path),device
    )

    #读取保存时记录的验证MSE
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    expected_mse = checkpoint["val_mse"]

    # 复用验证入口，不进行训练
    actual_mse = evaluate_main(encoder, decoder, device)

    assert abs(actual_mse - expected_mse) < 1e-6, (
        f"验证MSE不一致：保存时={expected_mse}，"
        f"加载后={actual_mse}"
    )

    print("检查点加载与验证指标一致性检查通过")

#-----------------5.多轮训练与验证-----------------------
def train_multiple_epochs(epochs):
    """
    从头训练多轮，每轮验证，并保存最佳验证权重。

    参数：
        epochs：完整遍历训练集的次数，默认为5

    返回：
        history：每轮训练与验证MSE的记录列表
    """
    if epochs <= 0:
        raise ValueError("训练轮数必须大于0")

    torch.manual_seed(24)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    project_dir = Path(__file__).resolve().parent.parent

    train_dataset = VideoFolderDataset(
        video_dir=project_dir / "data" / "videos" / "train",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    val_dataset = VideoFolderDataset(
        video_dir=project_dir / "data" / "videos" / "val",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    # 单独固定训练数据打乱所用的随机数生成器
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

    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    best_val_mse = float("inf")
    history = []

    save_path = (
        project_dir / "outputs" / "video_cnn_best_v1.pt"
    )

    print("运行设备：", device)

    for epoch in range(1, epochs + 1):
        print(f"\n===== 第 {epoch}/{epochs} 轮 =====")

        train_mse = train_one_epoch(
            encoder, decoder, train_loader, optimizer, device
        )

        val_mse = evaluate(
            encoder, decoder, val_loader, device
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse
        })

        print(
            f"第{epoch}轮："
            f"训练MSE={train_mse:.6f}，"
            f"验证MSE={val_mse:.6f}"
        )

        # 仅在验证误差降低时保存
        if val_mse < best_val_mse:
            best_val_mse = val_mse

            save_training_checkpoint(
                encoder=encoder,
                decoder=decoder,
                device=device,
                epoch=epoch,
                train_mse=train_mse,
                val_mse=val_mse,
                save_path=save_path
            )

    print(f"\n最佳验证MSE：{best_val_mse:.6f}")

    return history


def train_main():
    torch.manual_seed(24)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    project_dir = Path(__file__).resolve().parent.parent

    #创建数据集，每个样本包含4帧
    train_dataset = VideoFolderDataset(
        video_dir=project_dir / "data" / "videos" / "train",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,     # 每轮打乱GOP顺序
        num_workers=0,
        drop_last=False
    )

     # 创建新模型，从头训练
    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)

    # 同时管理编码器和解码器的参数
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )

    print("运行设备：", device)
    print("训练GOP数：", len(train_dataset))

    epoch_mse = train_one_epoch(
        encoder=encoder,
        decoder=decoder,
        loader=train_loader,
        optimizer=optimizer,
        device=device
    )

    print(f"第1轮平均训练MSE：{epoch_mse:.6f}")
    return encoder, decoder, device, epoch_mse

def evaluate_main(encoder, decoder, device):
    torch.manual_seed(24)
    
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    project_dir = Path(__file__).resolve().parent.parent
    
    #创建验证集，每个样本包含4帧
    val_dataset = VideoFolderDataset(
        video_dir=project_dir / "data" / "videos" / "val",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    val_mse = evaluate(
        encoder=encoder,
        decoder=decoder,
        loader=val_loader,
        device=device
    )

    print(f"第1轮验证MSE：{val_mse:.6f}")
    return val_mse

if __name__=='__main__':
    """
    encoder, decoder, device, train_mse = train_main()  # 接收训练结果
    val_mse = evaluate_main(encoder, decoder, device)  # 传给评测函数
    project_dir = Path(__file__).resolve().parent.parent

    save_training_checkpoint(
        encoder=encoder,
        decoder=decoder,
        device=device,
        epoch=1,
        train_mse=train_mse,
        val_mse=val_mse,
        save_path=(
            project_dir / "outputs" / "video_cnn_epoch1_v1.pt"
        )
    )
    """
    #check_saved_model()
    history = train_multiple_epochs(epochs=5)