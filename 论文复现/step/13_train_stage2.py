"""阶段II学习实验：冻结发送端，训练恢复网络和接收端；运行会更新参数。

训练PL=[0,.3,.6,.9]；验证PL=[0,.25,.5,.9]，固定通道seed42。
验证四条件MSE等权均值选best；不是论文明确给出的选择标准。
恢复目标为量化后特征，gamma默认0.1为学习配置。
"""
import argparse
import csv
from datetime import datetime
import importlib
import math
from pathlib import Path
import random
import shutil
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

s2=importlib.import_module('13_stage2_model')
multi=importlib.import_module('12_train_multi_gop')
utils,data,base=multi.utils,multi.data,multi.base
ROOT=Path(__file__).resolve().parent.parent
PROBABILITIES=(0.,.25,.5,.9)


@torch.no_grad()
def evaluate(model,dataset,device,output,epoch):
    """固定每帧通道随机源；同seed与Step13基线采用相同包掩码。"""
    model.eval()
    rows=[]
    images=[]
    middle=len(dataset)//2
    for probability in PROBABILITIES:
        error_sum,pixels,lost,total=0.,0,0,0
        for index in range(len(dataset)):
            item=dataset[index]
            video=item['video'].unsqueeze(0).to(device)
            mask=item['valid_mask'].unsqueeze(0).to(device)
            # Stage2内部会加branch*1000003，与基线对应。
            seed=42+item['start_frame']*1009
            pred,_,_,_=model(video,[probability],[seed])
            error_sum+=torch.where(mask,(pred-video).square(),0.).double().sum().item()
            pixels+=mask.expand_as(video).sum().item()
            for branch in model.branches:
                lost+=branch['packet_masks'].sum().item()
                total+=branch['packet_masks'].numel()
            if index==middle:
                if probability==0: images=[('Original',video.cpu())]
                images.append((f'p={probability:.0%}',pred.cpu()))
        mse=error_sum/pixels
        if not math.isfinite(mse): raise RuntimeError('验证MSE非有限')
        rows.append(dict(epoch=epoch,p=probability,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)),actual_loss=lost/total))
    item=dataset[middle]
    coords=item['valid_mask'][0,0].nonzero()
    top,left=coords.min(0).values.tolist();bottom,right=(coords.max(0).values+1).tolist()
    fig,axes=multi.plt.subplots(5,4,figsize=(10,12))
    for r,(label,tensor) in enumerate(images):
        for t in range(4):
            axes[r,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
            axes[r,t].set_title(f'{label} / frame {item["start_frame"]+t}')
            axes[r,t].axis('off')
    fig.suptitle(f'Stage II / epoch {epoch} / fixed validation channel seed 42')
    fig.tight_layout();fig.savefig(output/f'val_epoch_{epoch:03d}.png',dpi=130);multi.plt.close(fig)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--aux-weight',type=float,default=.01)
    parser.add_argument('--feature-weight',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--no-recovery',action='store_true',help='独立接收端训练对照：不执行恢复网络，恢复损失不参与优化')
    parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/step13_stage2_runs')
    args=parser.parse_args()
    if args.epochs<1 or not math.isfinite(args.lr) or args.lr<=0: parser.error('轮数和学习率必须为正')
    if any(not math.isfinite(w) or w<0 for w in (args.aux_weight,args.feature_weight)): parser.error('损失权重必须非负有限')
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['config'].get('model_variant','legacy')!='common_fig7': parser.error('当前对照固定使用common_fig7阶段I权重')
    train=data.CachedGOPDataset(args.data_dir,'train');val=data.CachedGOPDataset(args.data_dir,'val')
    if utils.sha256(args.data_dir/'manifest.json')!=saved['config']['manifest_sha256']: raise ValueError('数据清单不同')
    if utils.sha256(args.data_dir/'frames_uint8.npy')!=train.manifest['cache_sha256']: raise ValueError('缓存哈希不同')
    if not np.array_equal(np.load(args.data_dir/'valid_mask.npy'),np.load(args.checkpoint.parent/'valid_mask.npy')): raise ValueError('mask不同')
    codec=s2.variants.build_model('common_fig7');codec.load_state_dict(saved['model'],strict=True)
    model=s2.Stage2Model(codec,recovery_enabled=not args.no_recovery).to(args.device)
    effective_feature_weight=0. if args.no_recovery else args.feature_weight
    if args.smoke:
        train=Subset(train,range(min(2,len(train))));val=Subset(val,range(min(2,len(val))))
    generator=torch.Generator().manual_seed(args.seed)
    loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=0,generator=generator)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(stage1_checkpoint_sha256=utils.sha256(args.checkpoint),variant='common_fig7',
        recovery_enabled=not args.no_recovery,effective_feature_weight=effective_feature_weight,
        train_gops=len(train),val_gops=len(val),batch_size=1,packet_size=16,train_probabilities=[0,.3,.6,.9],
        validation_probabilities=PROBABILITIES,validation_seed=42,frozen_modules=list(model.frozen_names),
        quantization='eval rounding',feature_target='quantized transmitted features',
        feature_reduction='sum of whole-branch MSE means',selection='equal mean of four validation MSEs',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    utils.write_json(out/'config.json',config)
    utils.write_json(out/'stage1_config.json',saved['config'])
    shutil.copy2(args.data_dir/'manifest.json',out/'data_manifest.json')
    utils.write_json(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,python=multi.platform.python_version()))
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    started=time.perf_counter();updates=0;best=float('inf');best_epoch=0;history=[]
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
    def checkpoint(name,epoch,rows):
        temp=out/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
            config=config,metrics=rows,torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),temp)
        temp.replace(out/name)
    print(f'实验目录：{out}；阶段II将更新参数；恢复网络启用={model.recovery_enabled}；有效特征损失权重={effective_feature_weight}',flush=True)
    with (out/'updates.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['epoch','update','start_frame','p','channel_seed','mse','aux','feature_loss','total','grad_norm'])
        writer.writeheader()
        for epoch in range(args.epochs+1):
            if epoch:
                model.train()
                for batch in loader:
                    probability=random.choice([0.,.3,.6,.9]);seed=random.randrange(2**31)
                    video=batch['video'].to(args.device);mask=batch['valid_mask'].to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    pred,aux,features,_=model(video,[probability],[seed])
                    mse=base.masked_mse(pred,video,mask)
                    # 无恢复时，发送端被冻结，置零特征MSE没有可训练梯度。
                    # 明确置其权重为0，避免把常数项作为有效优化目标报告。
                    loss=mse+args.aux_weight*aux+effective_feature_weight*features
                    if not torch.isfinite(loss): raise RuntimeError('非有限训练损失')
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    optimizer.step();updates+=1
                    writer.writerow(dict(epoch=epoch,update=updates,start_frame=int(batch['start_frame'][0]),
                        p=probability,channel_seed=seed,mse=mse.item(),aux=aux.item(),feature_loss=features.item(),total=loss.item(),grad_norm=norm.item()));stream.flush()
                    del pred,aux,features,mse,loss
                    if updates%25==0: print(f'epoch={epoch} updates={updates}',flush=True)
            rows=evaluate(model,val,args.device,out,epoch)
            if epoch==0 and not args.smoke and abs(rows[0]['mse']-saved['metrics']['val_rounded_mse'])>1e-6:
                raise RuntimeError('无丢包起点未复现阶段I指标，停止训练')
            history.extend(rows)
            with (out/'validation.csv').open('w',newline='',encoding='utf-8') as vf:
                vw=csv.DictWriter(vf,fieldnames=list(rows[0]));vw.writeheader();vw.writerows(history)
            selection=sum(r['mse'] for r in rows)/4
            if selection<best: best,best_epoch=selection,epoch;checkpoint('best.pt',epoch,rows)
            checkpoint('last.pt',epoch,rows)
            assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in model.named_parameters() if not p.requires_grad)
            print(f'epoch={epoch} '+ ' | '.join('p={:.0%}: {:.3f} dB'.format(r['p'],r['psnr_db']) for r in rows),flush=True)
            report=dict(status='completed' if epoch==args.epochs else 'running',smoke=args.smoke,epoch=epoch,
                optimizer_updates=updates,best_mean_val_mse=best,best_epoch=best_epoch,final=rows,
                frozen_parameters_unchanged=True,elapsed_seconds=time.perf_counter()-started,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None)
            utils.write_json(out/'result.json',report)
    (out/'summary.txt').write_text('阶段II学习实验；发送端冻结，接收端和恢复网络联合训练。\n'
        f'恢复网络启用={model.recovery_enabled}；有效特征损失权重={effective_feature_weight}。无恢复模式只更新接收端，恢复网络参数冻结且不执行。\n'
        '损失=MSE+0.01*aux+0.1*特征恢复损失（以config实际权重为准，非论文核实系数）。\n'
        '验证固定seed42，四条件MSE等权均值选择best，last为最后一轮；需同时观察无丢包质量退化。\n'
        '新模型检查点包含恢复网络，不可直接使用阶段I评估脚本加载。尚未提供续训入口。\n'
        '同视频验证非独立测试；真实比特打包/UDP未实现。恢复目标为量化后特征，属于实验选择。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print(f'记录保存完成：{out}')


if __name__=='__main__': main()
