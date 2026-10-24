"""加载权重进行评测，保存CSV与原图/重建对照图，不调用训练。"""
import argparse
import csv
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hier_video.data.video import load_video_as_gop
from hier_video.checkpoint import load_models

def evaluate_gop(encoder, decoder, gop, image_path):
    """输入BTCHW视频；返回更新后模型的平均MSE并保存对照图。"""
    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        inputs = gop.to(device)
        outputs = decoder(encoder(inputs))
        mse = (outputs-inputs).square().mean().item()
    fig, axes = plt.subplots(2, inputs.shape[1], figsize=(3*inputs.shape[1],6), squeeze=False)
    for t in range(inputs.shape[1]):
        for row, tensor in enumerate([inputs, outputs]):
            axes[row,t].imshow(tensor[0,t].cpu().permute(1,2,0))
            axes[row,t].set_title(f'{"Original" if row == 0 else "Reconstructed"} {t+1}')
            axes[row,t].axis('off')
    fig.tight_layout()
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return mse

def main():
    parser=argparse.ArgumentParser(description='单视频分片评测；非独立测试集结论')
    parser.add_argument('--video', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--start-frames', type=int, nargs='+', default=[0,100])
    parser.add_argument('--num-frames', type=int, default=4)
    parser.add_argument('--crop-size', type=int, default=128)
    parser.add_argument('--output-dir', default='outputs/evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    encoder,decoder=load_models(args.checkpoint,args.device)
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for start in args.start_frames:
        gop=load_video_as_gop(args.video,args.crop_size,args.num_frames,start)
        mse=evaluate_gop(encoder,decoder,gop,output/f'frames_{start}.png')
        records.append(dict(start_frame=start,num_frames=args.num_frames,mse=mse,video=args.video,checkpoint=args.checkpoint))
        print(f'起始帧={start}, MSE={mse:.6f}')
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

if __name__=='__main__':
    main()
