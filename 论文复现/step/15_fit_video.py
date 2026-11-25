"""单视频多GOP拟合，按阶段增加样本，尚非泛化测试。

默认均匀选8个完整GOP（包括开头和结尾）；顺序解码一次，避免seek偏差。
训练500轮=每个GOP访问500次；每轮洗牌。--gops 0使用全部不重叠GOP。
缓存按GOP存uint8，训练才转float；输出包含逐段评估和固定预览。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import random
import shutil
import time
import subprocess
import sys

import cv2
import numpy as np
import torch

variants=importlib.import_module('12_model_variants')
base=variants.base
utils=importlib.import_module('12_train_single_gop')
ROOT=Path(__file__).resolve().parent.parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--gops',type=int,default=8,help='0表示全部完整GOP')
    p.add_argument('--model-variant',choices=('common_fig7','spatial_only','spatiotemporal_only','branches_float'),default='common_fig7')
    p.add_argument('--epochs',type=int,default=500,help='累计目标轮数')
    p.add_argument('--resume',type=Path,help='本脚本生成的last.pt；恢复Adam、随机状态和累计轮数')
    p.add_argument('--init-checkpoint',type=Path,help='仅加载模型权重；允许扩大GOP集合，Adam和轮数重新开始')
    p.add_argument('--transfer-codec',action='store_true',help='显式从spatiotemporal_only迁移四个模块到branches_float')
    p.add_argument('--enable-quantization',action='store_true',help='仅初始化：branches_float权重严格迁移到common_fig7，启用原3-bit量化，新建Adam')
    p.add_argument('--export-epochs',type=int,nargs='*',default=[],help='指定轮末自动导出整段视频，例如1 5 10')
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--allow-lr-change',action='store_true',help='显式允许续训改变学习率，保留Adam动量')
    p.add_argument('--quality-threshold',type=float,default=27.,help='当前视频暂定逐GOP验收线，非通用标准')
    p.add_argument('--accumulation-steps',type=int,default=1,help='平均若干GOP的梯度后做一次Adam更新')
    p.add_argument('--allow-accumulation-change',action='store_true',help='明确允许恢复时改变累积数，保留Adam状态')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/step15_video_fit')
    args=p.parse_args()
    if args.gops<0 or args.epochs<1 or args.eval_every<1:p.error('参数范围错误')
    if args.accumulation_steps<1:p.error('梯度累积数必须为正')
    if args.resume and args.init_checkpoint:p.error('resume和init-checkpoint互斥')
    if args.transfer_codec and (not args.init_checkpoint or args.model_variant!='branches_float'):
        p.error('transfer-codec需要init-checkpoint和branches_float')
    if args.enable_quantization and (not args.init_checkpoint or args.model_variant!='common_fig7' or args.transfer_codec):
        p.error('enable-quantization需要init-checkpoint和common_fig7，不能与transfer-codec一起使用')
    if not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.quality_threshold):p.error('学习率或验收阈值不合法')
    torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError('无法打开视频')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    all_starts=list(range(0,count-3,4))
    if not all_starts:raise RuntimeError('不足4帧')
    if args.gops>len(all_starts):p.error('请求GOP数量超过视频长度')
    starts=all_starts if args.gops==0 else [all_starts[i] for i in np.linspace(0,len(all_starts)-1,args.gops,dtype=int)]
    if args.gops==0 and count%4:starts.append((count//4)*4)
    actual_frames={s:min(4,count-s) for s in starts}
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    cache=out/'gops';cache.mkdir()
    needed={f for s in starts for f in range(s,s+4)}
    pending=[];decoded=0;info=None;mask=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if decoded in needed:
                pending.append(frame)
                if len(pending)==4:
                    video,current_mask,current_info=base.step05.frames_to_full_gop(pending,256)
                    if info is not None and current_info!=info:raise RuntimeError('视频尺寸变化')
                    info,mask=current_info,current_mask
                    np.save(cache/f'{decoded-3:06d}.npy',video[0].mul(255).round().byte().numpy())
                    pending=[]
            decoded+=1
    finally:cap.release()
    if pending:
        # 仅全量模式会包含末尾不足4帧的片段。
        start=count-len(pending)
        video,current_mask,current_info=base.step05.frames_to_full_gop(pending+[pending[-1]]*(4-len(pending)),256)
        if current_info!=info:raise RuntimeError('尾帧尺寸不同')
        np.save(cache/f'{start:06d}.npy',video[0].mul(255).round().byte().numpy())
    if decoded!=count:raise RuntimeError('实际帧数与元数据不同，停止以避免错误划分')
    assert len(list(cache.glob('*.npy')))==len(starts)
    torch.save(mask,out/'valid_mask.pt')
    def load(start):
        video=torch.from_numpy(np.load(cache/f'{start:06d}.npy')).float().div_(255).unsqueeze(0)
        return video.to(args.device)
    mask=mask.to(args.device)
    def sample_mask(start):
        # 新增时间mask：补齐帧不参与像素损失和指标。
        valid=mask.expand(1,4,1,256,256).clone()
        valid[:,actual_frames[start]:]=False
        return valid
    model=variants.build_model(args.model_variant).to(args.device)
    aux_weight=0. if args.model_variant=='spatial_only' else .01
    initialization_starts=[]
    transfer_report=None
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)
        if initial['config']['video_sha256']!=utils.sha256(args.video):p.error('初始化权重不是当前视频实验')
        if args.enable_quantization:
            if initial['config']['model_variant']!='branches_float':p.error('启用量化的迁移源须为branches_float')
            transfer_report=variants.transfer_float_branches_to_quantized(model,initial['model'])
        elif args.transfer_codec:
            if initial['config']['model_variant']!='spatiotemporal_only':p.error('迁移源须为spatiotemporal_only')
            transfer_report=variants.transfer_spatiotemporal_codec(model,initial['model'])
        else:
            if initial['config']['model_variant']!=args.model_variant:p.error('初始化模型结构不同')
            model.load_state_dict(initial['model'],strict=True)
        initialization_starts=initial['config']['gop_starts']
        del initial
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    previous_lr=None
    start_epoch=0;updates=0;saved_metrics=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        old=saved['config']
        initialization_starts=old.get('initialization_starts',[])
        for key,value in (('seed',args.seed),('video_sha256',utils.sha256(args.video)),
                          ('gop_starts',starts),('model_variant',args.model_variant)):
            if old[key]!=value:p.error(f'续训配置不一致：{key}')
        if not torch.equal(torch.load(args.resume.parent/'valid_mask.pt',map_location='cpu',weights_only=False),mask.cpu()):
            p.error('有效区域mask不同')
        for start in starts:
            if not np.array_equal(np.load(cache/f'{start:06d}.npy'),np.load(args.resume.parent/'gops'/f'{start:06d}.npy')):
                p.error(f'第{start}帧GOP缓存与原实验不同')
        start_epoch=int(saved['epoch']);updates=int(saved['updates'])
        if args.epochs<=start_epoch:p.error('累计目标epochs必须大于检查点轮数')
        previous_accumulation=old.get('accumulation_steps',1)
        if previous_accumulation!=args.accumulation_steps and not args.allow_accumulation_change:
            p.error('梯度累积数改变需添加--allow-accumulation-change')
        expected_old=old.get('start_updates',0)+(start_epoch-old.get('start_epoch',0))*math.ceil(len(starts)/previous_accumulation)
        if updates!=expected_old:p.error('检查点更新次数与完整轮次不一致')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        previous_lr=float(optimizer.param_groups[0]['lr'])
        if previous_lr!=args.lr and not args.allow_lr_change:
            p.error('学习率与检查点不同；若有意调整请添加--allow-lr-change')
        # load_state_dict会恢复旧lr，须在其后更新；保留所有Adam动量及计数。
        for group in optimizer.param_groups:group['lr']=args.lr
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if args.device=='cuda' and saved['cuda_rng'] is not None:
            if len(saved['cuda_rng'])!=torch.cuda.device_count():p.error('可见GPU数量与保存时不同')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        saved_metrics=saved['metrics']
        del saved
    start_updates=updates
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(video_sha256=utils.sha256(args.video),fps=fps,decoded_frames=decoded,
        initialization='weights_only' if args.init_checkpoint else ('resume' if args.resume else 'random'),
        initialization_starts=initialization_starts,actual_frames_per_gop=actual_frames,
        init_checkpoint_sha256=utils.sha256(args.init_checkpoint) if args.init_checkpoint else None,
        start_epoch=start_epoch,start_updates=start_updates,
        parent_checkpoint_sha256=utils.sha256(args.resume) if args.resume else None,
        gop_starts=starts,unused_tail_frames=0 if args.gops==0 else count%4,preprocessing=info,model_variant=args.model_variant,
        lr=args.lr,previous_lr=previous_lr,lr_changed=previous_lr is not None and previous_lr!=args.lr,
        aux_weight=aux_weight,batch_size=1,effective_gops_per_update=args.accumulation_steps,
        updates_per_epoch=math.ceil(len(starts)/args.accumulation_steps),
        expected_updates=updates+(args.epochs-start_epoch)*math.ceil(len(starts)/args.accumulation_steps),
        purpose='same-video training-set fit; no held-out evaluation',
        quantization_bits=3 if args.model_variant=='common_fig7' else None,
        parameters=sum(q.numel() for q in model.parameters()),
        log_placeholder_fields=(['eval_aux'] if args.model_variant=='spatial_only' else [])+
            (['eval_common_clip','eval_individual_clip'] if args.model_variant in ('spatial_only','spatiotemporal_only') else []),
        note='sampling/scene cuts and duplicate animation frames retained; not all-video fit until gops=0')
    utils.write_json(out/'config.json',config)
    if transfer_report:utils.write_json(out/'weight_transfer.json',transfer_report)
    if args.resume:
        parent=out/'parent_records';parent.mkdir()
        for name in ('config.json','result.json','history.json','per_gop.csv','updates.csv','summary.txt'):
            source=args.resume.parent/name
            if source.exists():shutil.copy2(source,parent/name)
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(Path(base.step05.__file__),out/'source/step/05_video_dataset.py')
    utils.write_json(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None))
    show=starts if len(starts)<=8 else [starts[0],starts[len(starts)//2],starts[-1]]
    before={}
    for s in show:
        before[s]=utils.evaluate(model,load(s),sample_mask(s))[0]
    best=float('inf');best_epoch=start_epoch;history=[];started=time.perf_counter()
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    def checkpoint(name,epoch,metrics):
        temporary=out/(name+'.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
            config=config,metrics=metrics,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device=='cuda' else None),temporary)
        temporary.replace(out/name)
    print(f'实验目录：{out}；GOP数={len(starts)}；更新数={config["expected_updates"]}',flush=True)
    print(f'固定起始帧：{starts}',flush=True)
    print(f'累计轮数：{start_epoch} → {args.epochs}；起点更新={start_updates}；新增更新={config["expected_updates"]-start_updates}',flush=True)
    print(f'学习率：{previous_lr} → {args.lr}；验收阈值={args.quality_threshold} dB',flush=True)
    print(f'模型={args.model_variant}，参数量={config["parameters"]}，辅助损失权重={aux_weight}',flush=True)
    if transfer_report:print(f'权重迁移：{transfer_report}',flush=True)
    print(f'梯度累积：{args.accumulation_steps}个GOP；每轮Adam更新={config["updates_per_epoch"]}',flush=True)
    with (out/'updates.csv').open('w',newline='',encoding='utf-8') as stream, (out/'per_gop.csv').open('w',newline='',encoding='utf-8') as detail:
        writer=csv.DictWriter(stream,fieldnames=['epoch','update','start_frame','mse','aux','total','grad_norm']);writer.writeheader()
        dw=None
        for epoch in range(start_epoch,args.epochs+1):
            if epoch>start_epoch:
                order=starts.copy();random.shuffle(order)
                model.train()
                for group_start in range(0,len(order),args.accumulation_steps):
                    group=order[group_start:group_start+args.accumulation_steps]
                    optimizer.zero_grad(set_to_none=True)
                    group_logs=[]
                    for start in group:
                        video=load(start)
                        pred,aux,_=model(video);mse=base.masked_mse(pred,video,sample_mask(start));loss=mse+aux_weight*aux
                        if not torch.isfinite(loss):raise RuntimeError('非有限loss')
                        # 末组可能不足4个，按实际组大小平均，避免末组梯度偏小。
                        # 每个GOP的MoE统计保持原定义，不等价于跨4GOP联合统计负载。
                        (loss/len(group)).backward()
                        group_logs.append(dict(epoch=epoch,start_frame=start,mse=mse.item(),aux=aux.item(),total=loss.item()))
                        del pred,aux,mse,loss
                    # 累积完成后只裁剪一次、只更新一次Adam。
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();updates+=1
                    for entry in group_logs:writer.writerow(dict(**entry,update=updates,grad_norm=norm.item()))
                    stream.flush()
                    if updates%100==0:print(f'epoch={epoch}/{args.epochs} updates={updates}',flush=True)
                if epoch%10==0:print(f'epoch={epoch}/{args.epochs} updates={updates}',flush=True)
            if epoch!=start_epoch and epoch%args.eval_every!=0 and epoch!=args.epochs and epoch not in args.export_epochs:continue
            metrics=[]
            for start in starts:
                video=load(start);after,result=utils.evaluate(model,video,sample_mask(start))
                row=dict(epoch=epoch,start_frame=start,**result);metrics.append(row)
                if dw is None:dw=csv.DictWriter(detail,fieldnames=list(row));dw.writeheader()
                dw.writerow(row)
                # 8-GOP阶段每段保存；全视频时仅固定三段避免成千上万图片。
                if start in show:
                    utils.preview(out/f'epoch_{epoch:04d}_frame_{start:06d}.png',video,before[start],after,start,updates,mask,start_updates)
            detail.flush()
            pooled=sum(r['eval_mse']*actual_frames[r['start_frame']] for r in metrics)/sum(actual_frames.values())
            if epoch==start_epoch and saved_metrics is not None and abs(pooled-saved_metrics['mean_mse'])>1e-6:
                raise RuntimeError('恢复起点MSE与检查点不一致，停止训练')
            summary=dict(epoch=epoch,updates=updates,mean_mse=pooled,pooled_psnr_db=-10*math.log10(max(pooled,1e-12)),
                worst_gop_psnr_db=min(r['eval_psnr_db'] for r in metrics),elapsed_seconds=time.perf_counter()-started)
            accepted=[r for r in metrics if r['eval_psnr_db']>=args.quality_threshold]
            summary.update(quality_threshold=args.quality_threshold,passed_gops=len(accepted),total_gops=len(metrics),
                gop_pass_rate=len(accepted)/len(metrics),
                frame_coverage_rate=sum(actual_frames[r['start_frame']] for r in accepted)/sum(actual_frames.values()))
            for label,subset in (('initialization_gops',[r for r in metrics if r['start_frame'] in initialization_starts]),
                                 ('new_gops',[r for r in metrics if r['start_frame'] not in initialization_starts])):
                if subset:
                    value=sum(r['eval_mse']*actual_frames[r['start_frame']] for r in subset)/sum(actual_frames[r['start_frame']] for r in subset)
                    summary[label+'_psnr_db']=-10*math.log10(max(value,1e-12))
            history.append(summary)
            if pooled<best:best,best_epoch=pooled,epoch;checkpoint('best.pt',epoch,summary)
            checkpoint('last.pt',epoch,summary)
            utils.write_json(out/'history.json',history)
            utils.write_json(out/'result.json',dict(status='completed' if epoch==args.epochs else 'running',final=summary,
                best_mse=best,best_epoch=best_epoch,samples_per_epoch=len(starts),visits_per_gop=epoch,
                start_epoch=start_epoch,updates_this_run=updates-start_updates,best_scope='current run including resumed start',
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None))
            fig,ax=base.plt.subplots(figsize=(7,4));ax.plot([r['epoch'] for r in history],[r['pooled_psnr_db'] for r in history],'o-',label='Pooled PSNR')
            ax.plot([r['epoch'] for r in history],[r['worst_gop_psnr_db'] for r in history],'o-',label='Worst GOP')
            ax.set(xlabel='Epoch / visits per GOP',ylabel='Rounded PSNR (dB)');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);base.plt.close(fig)
            print(f'epoch={epoch} 汇总PSNR={summary["pooled_psnr_db"]:.3f} 最差GOP={summary["worst_gop_psnr_db"]:.3f} 达标={len(accepted)}/{len(metrics)} ({summary["gop_pass_rate"]:.2%})',flush=True)
            if epoch in args.export_epochs:
                # 导出使用独立检查点，之后last.pt被覆盖也不丢失对应权重。
                name=f'epoch_{epoch:04d}.pt';checkpoint(name,epoch,summary)
                command=[sys.executable,'-u','-B',str(Path(__file__).parent/'16_reconstruct_video.py'),
                    '--checkpoint',str(out/name),'--video',str(args.video),'--device',args.device,
                    '--output-root',str(out/'video_exports'/f'epoch_{epoch:04d}')]
                print(f'导出第{epoch}轮完整视频',flush=True)
                subprocess.run(command,check=True)
    (out/'summary.txt').write_text('单视频训练集拟合诊断，非泛化测试。默认8个GOP，不代表整段已重建。\n'
        '每轮每样本访问一次；记录平均和最差片段指标，帧号预先固定。\n'
        f'模型版本={args.model_variant}。本次起点{start_epoch}轮，新增{updates-start_updates}次更新。\n'
        'spatial_only只有逐帧空间编解码，无时空压缩/双分支/MoE/量化，辅助项和裁剪值为0占位；不是同码率比较。\n'
        'spatiotemporal_only增加两级时空编解码与8个MoE块，损失为MSE+0.01辅助损失，无分支分解和量化；裁剪统计为0占位。\n'
        'branches_float加入公共/个体压缩和上采样，无任何量化或裁剪。日志超范围比例仅诊断相对[-1,1]的分布，不表示执行了裁剪。\n'
        'transfer-codec仅迁移四个编解码模块，新分支随机初始化，新Adam，所有模块联合训练；起点并非恒等于源模型。\n'
        '支持--resume恢复；--epochs指定累计目标。best与曲线范围为本次运行含起点，原记录在parent_records。\n'
        f'学习率由{previous_lr}变为{args.lr}；保留Adam状态。验收阈值{args.quality_threshold} dB仅用于当前内容，达标率不代替播放验收。\n'
        f'每{args.accumulation_steps}个GOP梯度平均后更新一次，末组按实际数量平均；MoE负载损失仍逐GOP计算。每轮样本访问次数不变，Adam更新次数改变。\n'
        'init-checkpoint仅恢复模型，新Adam；与同样本集合resume不同。全量模式包含尾部GOP，重复补齐帧不参与损失。\n'
        'export-epochs调用整段重建脚本并保存对应轮权重。实际导出目录见video_exports。\n',encoding='utf-8')
    print(f'训练记录保存完成：{out}')


if __name__=='__main__':main()
