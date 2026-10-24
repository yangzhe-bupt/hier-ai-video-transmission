import torch
from pathlib import Path
from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.models.semantic_decoder import GOPDecoder

def save_checkpoint(encoder, decoder, losses, save_path):
    """保存编解码器参数和训练损失。"""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": "lightweight_cnn_v1",
        "feature_channels": 24,
        "layout": "BTCHW",
        "training_scope": "single_gop",
        "steps": len(losses),
        "encoder": {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()},
        "decoder": {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()},
        "losses": losses
    }
    torch.save(checkpoint, path)
    print("检查点已保存：", path.resolve())


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

