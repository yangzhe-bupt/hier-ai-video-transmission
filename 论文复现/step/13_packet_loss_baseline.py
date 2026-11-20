"""无恢复网络的整包擦除基线；冻结权重，只评估验证集。

通过量化器输出hook接入通道，避免重新抄写编解码路径。
固定交织，公共/个体分支分别分包，每包16个量化元素。
这是元素张量上的整包擦除仿真，尚非真实3-bit比特打包或UDP传输。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import shutil
import time

import torch

multi = importlib.import_module('12_train_multi_gop')
variants = importlib.import_module('12_model_variants')
utils, data, base = multi.utils, multi.data, multi.base
ROOT = Path(__file__).resolve().parent.parent


class PacketChannel:
    """先交织、按包擦除、再逆交织。mask=True表示整包丢失。

    每个GOP/分支使用独立CPU生成器；同seed不同p共享随机数，
    使高丢包率掩码包含低丢包率掩码，便于比较。
    """
    def __init__(self, packet_size):
        self.packet_size = packet_size
        self.interleavers = {}
        self.records = []

    def apply(self, branch, features):
        assert features.shape[0] == 1
        n = features[0].numel()
        if branch not in self.interleavers:
            self.interleavers[branch] = base.step09.SemanticInterleaver(n, seed=42+branch)
        interleaver = self.interleavers[branch]
        packets = (n+self.packet_size-1)//self.packet_size
        generator = torch.Generator().manual_seed(self.seed + self.start_frame*1009 + branch*1000003)
        lost = torch.rand(packets, generator=generator) < self.probability
        # 最后一包可不满，按真实元素数截断；包丢失率和元素丢失率分别记录。
        mask = lost.repeat_interleave(self.packet_size)[:n].reshape(features.shape).to(features.device)
        received = interleaver(features).masked_fill(mask, 0)
        result = interleaver.deinterleave(received)
        self.records.append({'branch': branch, 'packets': packets, 'lost_packets': int(lost.sum()),
                             'elements': n, 'lost_elements': int(mask.sum()),
                             'packet_mask': lost})
        if self.probability == 0:
            assert torch.equal(result, features)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--packet-size', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/step13_packet_baseline')
    args = parser.parse_args()
    if args.packet_size < 1:
        parser.error('每包元素数必须为正')
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = variants.build_model(saved['config'].get('model_variant','legacy')).to(args.device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    dataset = data.CachedGOPDataset(args.data_dir,'val')
    if utils.sha256(args.data_dir/'manifest.json') != saved['config']['manifest_sha256']:
        raise ValueError('数据清单与训练时不同')
    if utils.sha256(args.data_dir/'frames_uint8.npy') != dataset.manifest['cache_sha256']:
        raise ValueError('缓存哈希不同')
    original_mask = torch.from_numpy(multi.np.load(args.checkpoint.parent/'valid_mask.npy'))
    if not torch.equal(dataset.mask, original_mask):
        raise ValueError('有效区域mask与训练时不同')
    indices = list(range(min(2,len(dataset)) if args.smoke else len(dataset)))
    channel = PacketChannel(args.packet_size)
    handles = [module.register_forward_hook(lambda m,i,o,b=b: channel.apply(b,o))
               for b,module in enumerate((model.common_quantizer,model.individual_quantizer))]
    output = args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    checkpoint_hash = utils.sha256(args.checkpoint)
    utils.write_json(output/'config.json', {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'checkpoint_sha256':checkpoint_hash, 'model_variant':saved['config'].get('model_variant','legacy'),
        'epoch':saved['epoch'], 'probabilities':[0,.25,.5,.9], 'optimizer_updates':0,
        'interleaver_seeds':[42,43], 'torch':torch.__version__,
        'note':'two separate branch packet streams; zero fill; no recovery; internal video black bars retained'})
    shutil.copy2(args.data_dir/'manifest.json',output/'data_manifest.json')
    source_hashes = {}
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            relative = source.relative_to(ROOT)
            target = output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            source_hashes[str(relative)] = utils.sha256(source)
    utils.write_json(output/'source_manifest.json',source_hashes)
    rows, masks = [], []
    started = time.perf_counter()
    baseline_mse = None
    print(f'输出目录：{output}；不训练',flush=True)
    with torch.no_grad():
        for seed in args.seeds:
            for probability in (0,.25,.5,.9):
                channel.seed, channel.probability = seed, probability
                error_sum, pixels, total_packets, lost_packets = 0.,0,0,0
                for index in indices:
                    sample = dataset[index]
                    channel.start_frame = sample['start_frame']
                    channel.records = []
                    video = sample['video'].unsqueeze(0).to(args.device)
                    valid = sample['valid_mask'].unsqueeze(0).to(args.device)
                    prediction = model(video)[0]
                    error_sum += torch.where(valid,(prediction-video).square(),0.).double().sum().item()
                    pixels += valid.expand_as(video).sum().item()
                    for record in channel.records:
                        total_packets += record['packets']
                        lost_packets += record['lost_packets']
                        masks.append({'seed':seed,'p':probability,'start_frame':channel.start_frame,**record})
                    # 同一验证GOP展示原图和四种条件，每个seed保存一张。
                    if index == indices[len(indices)//2]:
                        if probability == 0:
                            images = [('Original',video.cpu())]
                        images.append((f'p={probability:.0%}',prediction.cpu()))
                mse = error_sum/pixels
                if not math.isfinite(mse):
                    raise RuntimeError('非有限MSE')
                if probability == 0:
                    baseline_mse = mse
                    if not args.smoke and abs(mse-saved['metrics']['val_rounded_mse']) > 1e-6:
                        raise RuntimeError('0%丢包未复现检查点指标，停止其他条件评估')
                row = dict(seed=seed,p=probability,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),
                           packets=total_packets,lost_packets=lost_packets,actual_packet_loss=lost_packets/total_packets)
                rows.append(row)
                print(f'seed={seed} p={probability:.0%} 实际丢包={row["actual_packet_loss"]:.2%} MSE={mse:.8f} PSNR={row["psnr_db"]:.3f}',flush=True)
                with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
                if probability == .9:
                    fig,axes=multi.plt.subplots(5,4,figsize=(10,12))
                    positions=dataset[indices[len(indices)//2]]['valid_mask'][0,0].nonzero()
                    top,left=positions.min(0).values.tolist(); bottom,right=(positions.max(0).values+1).tolist()
                    for r,(label,tensor) in enumerate(images):
                        for t in range(4):
                            axes[r,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
                            axes[r,t].set_title(f'{label} / frame {dataset[indices[len(indices)//2]]["start_frame"]+t}')
                            axes[r,t].axis('off')
                    fig.tight_layout();fig.savefig(output/f'comparison_seed_{seed}.png',dpi=130);multi.plt.close(fig)
    for handle in handles: handle.remove()
    assert utils.sha256(args.checkpoint)==checkpoint_hash
    torch.save(masks,output/'packet_masks.pt')
    utils.write_json(output/'result.json',dict(status='completed',smoke=args.smoke,optimizer_updates=0,
        baseline_mse=baseline_mse,checkpoint_unchanged=True,elapsed_seconds=time.perf_counter()-started,metrics=rows))
    (output/'summary.txt').write_text('无恢复网络整包擦除基线。每包16元素（或config指定），公共/个体分别分包；交织后擦除，再逆交织。\n'
        '仅验证集评估；三随机种子为通道重复，不是三次独立模型训练。不同p共享随机数，mask嵌套。\n'
        'metrics.csv保存每次指标；packet_masks.pt保存实际包掩码。原视频内黑边仍参与指标。\n'
        '未执行参数更新，未做真实比特打包或UDP传输；不与论文90%丢包成绩直接比较。\n',encoding='utf-8')
    print(f'评估完成：{output}',flush=True)


if __name__=='__main__': main()
