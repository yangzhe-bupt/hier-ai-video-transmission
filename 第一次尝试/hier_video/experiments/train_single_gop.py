"""从仓库根目录运行：python -m hier_video.experiments.train_single_gop --help"""
import argparse
import torch
import torch.nn as nn
from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.models.semantic_decoder import GOPDecoder
from hier_video.data.video import load_video_as_gop
from hier_video.checkpoint import save_checkpoint

def train_single_gop(gop, steps, learning_rate, device="cpu"):
    """在固定GOP上训练轻量编解码器，返回两端模型和逐步损失。"""
    if steps <= 0 or learning_rate <= 0:
        raise ValueError("steps和learning_rate必须为正")
    torch.manual_seed(24)
    device = torch.device(device)
    gop = gop.to(device)

    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)
    encoder.train()
    decoder.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        features = encoder(gop)
        reconstructed = decoder(features)

        loss = criterion(reconstructed, gop)
        if not torch.isfinite(loss).item():
            raise RuntimeError("损失出现NaN或无穷值")

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step == 0 or (step + 1) % 20 == 0:
            print(f"步骤 {step + 1}/{steps}，MSE={loss.item():.6f}")

    print("训练设备：", device)
    return encoder, decoder, losses


def main():
    parser = argparse.ArgumentParser(description="固定GOP过拟合检查，不代表泛化性能")
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', default='outputs/single_gop_v1.pt')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--num-frames', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.crop_size <= 0 or args.crop_size % 4:
        parser.error('crop-size必须为正且能被4整除')
    from pathlib import Path
    if Path(args.checkpoint).exists():
        parser.error('检查点已存在，请使用新路径，避免覆盖已有实验')
    gop = load_video_as_gop(args.video, args.crop_size, args.num_frames, args.start_frame)
    encoder, decoder, losses = train_single_gop(gop, args.steps, args.learning_rate, args.device)
    save_checkpoint(encoder, decoder, losses, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    checkpoint['training_config'] = vars(args)
    checkpoint['seed'] = 24
    torch.save(checkpoint, args.checkpoint)

if __name__ == '__main__':
    main()
