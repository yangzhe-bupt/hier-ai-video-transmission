"""阶段II最佳权重评估与恢复模块绕过诊断；不更新参数。

固定seed42/43/44，与无恢复阶段I基线按同一个GOP/分支公式生成包掩码。
绕过模式保留阶段II解码器，只把恢复网络输出替换为收到的置零特征。
该推理诊断不等于重新训练的消融实验。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time

import torch

s2=importlib.import_module('13_stage2_model')
multi=importlib.import_module('12_train_multi_gop')
utils,data,plt=multi.utils,multi.data,multi.plt
ROOT=Path(__file__).resolve().parent.parent
PS=(0.,.25,.5,.9)


def draw_training_curves(source,output):
    with source.open(encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for p in PS:
        selected=[r for r in rows if float(r['p'])==p]
        for ax,key in zip(axes,('mse','psnr_db')):
            ax.plot([int(r['epoch']) for r in selected],[float(r[key]) for r in selected],'o-',label=f'p={p:.0%}')
            ax.set_xlabel('Stage II epoch');ax.set_ylabel(key);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,help='阶段I丢包基线目录，用于同seed数值对照')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--smoke',action='store_true',help='仅前2个GOP；不与全量基线比较')
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/step13_stage2_evaluation')
    args=parser.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    dataset=data.CachedGOPDataset(args.data_dir,'val')
    original_manifest=json.loads((args.checkpoint.parent/'data_manifest.json').read_text(encoding='utf-8'))
    # 允许Windows/Linux视频路径不同，实际内容和帧划分必须相同。
    for key in ('video_sha256','cache_sha256','preprocessing','splits','canvas_size','gop_length'):
        if dataset.manifest[key]!=original_manifest[key]: raise ValueError(f'数据清单与训练时不同：{key}')
    if utils.sha256(args.data_dir/'frames_uint8.npy')!=dataset.manifest['cache_sha256']: raise ValueError('缓存哈希不同')
    config=saved['config']
    codec=s2.variants.build_model(config['variant'])
    model=s2.Stage2Model(codec,packet_size=config['packet_size'],recovery_enabled=config.get('recovery_enabled',True)).to(args.device)
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    original={k:v.cpu().clone() for k,v in model.state_dict().items()}
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    checkpoint_hash=utils.sha256(args.checkpoint)
    utils.write_json(out/'config.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,
        epoch=saved['epoch'],device=args.device,torch=torch.__version__,smoke=args.smoke,
        recovery_enabled=config.get('recovery_enabled',True),
        seeds=[42,43,44],packet_size=config['packet_size'],optimizer_updates=0))
    for name in ('validation.csv','config.json','data_manifest.json'):
        shutil.copy2(args.checkpoint.parent/name,out/('training_'+name))
    draw_training_curves(args.checkpoint.parent/'validation.csv',out)
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    baseline=[]
    if args.baseline_dir:
        bc=json.loads((args.baseline_dir/'config.json').read_text(encoding='utf-8'))
        if bc['checkpoint_sha256']!=config['stage1_checkpoint_sha256'] or bc['packet_size']!=config['packet_size']:
            raise ValueError('阶段I基线权重或包尺寸与本实验不同')
        if json.loads((args.baseline_dir/'data_manifest.json').read_text(encoding='utf-8'))!=original_manifest:
            raise ValueError('阶段I基线数据不同')
        with (args.baseline_dir/'metrics.csv').open(encoding='utf-8') as stream: baseline=list(csv.DictReader(stream))
    indices=list(range(min(2,len(dataset)) if args.smoke else len(dataset)))
    rows=[];mask_hashes={};no_loss_outputs={};started=time.perf_counter()
    print(f'评估目录：{out}；checkpoint epoch={saved["epoch"]}；参数更新0次',flush=True)
    for mode in ('enabled','bypassed'):
        # 恢复网络本身仍执行，仅替换其输出；不改权重或state_dict结构。
        handles=[]
        if mode=='bypassed':
            handles=[m.register_forward_hook(lambda m,i,o:i[0]) for m in (model.common_recovery,model.individual_recovery)]
        with torch.no_grad():
            for seed in (42,43,44):
                preview_images={}
                for p in PS:
                    errors,pixels,lost,total=0.,0,0,0
                    digest=hashlib.sha256()
                    for index in indices:
                        item=dataset[index];video=item['video'].unsqueeze(0).to(args.device)
                        mask=item['valid_mask'].unsqueeze(0).to(args.device)
                        pred=model(video,[p],[seed+item['start_frame']*1009])[0]
                        if p==0 and seed==42:
                            if mode=='enabled': no_loss_outputs[index]=pred.cpu()
                            else: torch.testing.assert_close(pred.cpu(),no_loss_outputs[index],rtol=0,atol=0)
                        errors+=torch.where(mask,(pred-video).square(),0.).double().sum().item()
                        pixels+=mask.expand_as(video).sum().item()
                        for branch in model.branches:
                            pm=branch['packet_masks'];lost+=pm.sum().item();total+=pm.numel();digest.update(pm.numpy().tobytes())
                        if seed==42 and index==indices[len(indices)//2]:
                            preview_images[p]=(video.cpu(),pred.cpu(),item)
                    key=(seed,p)
                    if mode=='enabled': mask_hashes[key]=digest.hexdigest()
                    else: assert mask_hashes[key]==digest.hexdigest()
                    mse=errors/pixels
                    if not math.isfinite(mse): raise RuntimeError('非有限MSE')
                    row=dict(mode=mode,seed=seed,p=p,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),actual_loss=lost/total,mask_sha256=digest.hexdigest())
                    matches=[b for b in baseline if int(b['seed'])==seed and float(b['p'])==p]
                    if matches and not args.smoke:
                        b=matches[0]
                        assert abs(float(b['actual_packet_loss'])-row['actual_loss'])<1e-12
                        row['stage1_psnr_db']=float(b['psnr_db']);row['gain_vs_stage1_db']=row['psnr_db']-float(b['psnr_db'])
                    rows.append(row)
                    print(f'{mode} seed={seed} p={p:.0%} MSE={mse:.8f} PSNR={row["psnr_db"]:.3f}',flush=True)
                if seed==42:
                    fig,axes=plt.subplots(5,4,figsize=(10,12))
                    for r,p in enumerate((None,*PS)):
                        video,pred,item=preview_images[0 if p is None else p]
                        coords=item['valid_mask'][0,0].nonzero();top,left=coords.min(0).values.tolist();bottom,right=(coords.max(0).values+1).tolist()
                        tensor=video if p is None else pred
                        for t in range(4):
                            axes[r,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
                            axes[r,t].set_title(('Original' if p is None else f'{mode} p={p:.0%}')+f' / frame {item["start_frame"]+t}');axes[r,t].axis('off')
                    fig.tight_layout();fig.savefig(out/f'{mode}_seed42.png',dpi=130);plt.close(fig)
        for h in handles:h.remove()
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for mode in ('enabled','bypassed'):
        for p in PS:
            r=[r for r in rows if r['mode']==mode and r['p']==p]
            summary.append(dict(mode=mode,p=p,mean_mse=statistics.mean(x['mse'] for x in r),
                mean_psnr_db=statistics.mean(x['psnr_db'] for x in r),std_psnr_db=statistics.stdev(x['psnr_db'] for x in r)))
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    assert utils.sha256(args.checkpoint)==checkpoint_hash
    utils.write_json(out/'result.json',dict(status='completed',optimizer_updates=0,epoch=saved['epoch'],
        smoke=args.smoke,weights_unchanged=True,paired_packet_masks_identical=True,summary=summary,elapsed_seconds=time.perf_counter()-started))
    (out/'summary.txt').write_text('冻结阶段II模型，启用恢复/绕过恢复共享相同包掩码。\n'
        'mean_psnr_db是三个通道种子的PSNR算术均值，std为样本标准差；不是独立模型训练重复。\n'
        '绕过诊断仍使用阶段II适应后的解码器，不能替代重新训练消融。\n'
        '训练曲线来自原validation.csv，选模型的seed42属于验证条件；43/44是额外通道重复，不是跨视频测试。\n'
        '原视频内部黑边仍参与指标；未更新参数。\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'评估完成：{out}',flush=True)


if __name__=='__main__':main()
