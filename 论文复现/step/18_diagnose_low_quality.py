"""针对指定轮次未达标GOP的冻结量化诊断，不训练。

按分数低/中/接近阈值三组，各选确定性代表样本。选择仅依据原基线，
不是随机代表全视频；均值只描述所选样本。直通不是同码率方案。
"""
import argparse
import csv
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import shutil
import statistics
import zipfile

import numpy as np
import torch

variants=importlib.import_module('12_model_variants')
base=variants.base
utils=importlib.import_module('12_train_single_gop')
diagnostic=importlib.import_module('12_diagnose_quantization')
ROOT=Path(__file__).resolve().parent.parent


def select(rows,n):
    """按失败分数排序分三等份；组内按难度/中位/接近阈值取样。

    尽可能避免同组内相邻32帧，防止三个样本均来自同一瞬间。
    """
    rows=sorted(rows,key=lambda r:(r['psnr_db'],r['start_frame']))
    chunks=np.array_split(np.arange(len(rows)),3)
    chosen=[]
    for label,indices in zip(('low','middle','near_threshold'),chunks):
        group=[rows[int(i)] for i in indices]
        if label=='near_threshold':group=list(reversed(group))
        elif label=='middle':
            center=statistics.median(r['psnr_db'] for r in group)
            group=sorted(group,key=lambda r:(abs(r['psnr_db']-center),r['start_frame']))
        picked=[]
        for row in group:
            if all(abs(row['start_frame']-x['start_frame'])>=32 for x in picked):picked.append(row)
            if len(picked)==n:break
        chosen.extend(dict(**r,group=label) for r in picked)
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,help='默认run-dir/last.pt，CSV按权重实际epoch选择')
    p.add_argument('--threshold',type=float,default=25.)
    p.add_argument('--per-group',type=int,default=3)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/step18_quality_diagnosis')
    args=p.parse_args()
    if args.per_group<1 or not math.isfinite(args.threshold):p.error('参数错误')
    torch.set_num_threads(2)
    checkpoint=args.checkpoint or args.run_dir/'last.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=saved['config'];epoch=saved['epoch']
    with (args.run_dir/'per_gop.csv').open(encoding='utf-8') as f:
        rows=[dict(start_frame=int(r['start_frame']),psnr_db=float(r['eval_psnr_db']),mse=float(r['eval_mse']))
              for r in csv.DictReader(f) if int(r['epoch'])==epoch]
    assert len(rows)==len(config['gop_starts']) and {r['start_frame'] for r in rows}==set(config['gop_starts'])
    failed=[r for r in rows if r['psnr_db']<args.threshold]
    if len(failed)<3:raise ValueError('未达标样本不足三组')
    selected=select(failed,args.per_group)
    model=variants.build_model(config['model_variant']).to(args.device)
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    original_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    originals=(model.common_quantizer,model.individual_quantizer)
    model.common_quantizer=diagnostic.DiagnosticQuantizer(originals[0])
    model.individual_quantizer=diagnostic.DiagnosticQuantizer(originals[1])
    mask=torch.load(args.run_dir/'valid_mask.pt',map_location='cpu',weights_only=False)
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    checkpoint_hash=utils.sha256(checkpoint)
    utils.write_json(out/'selection.json',dict(epoch=epoch,threshold=args.threshold,selected=selected,
        selection_rule='failed score terciles, deterministic within groups; minimum32frame separation when possible',
        checkpoint_sha256=checkpoint_hash,csv_sha256=utils.sha256(args.run_dir/'per_gop.csv')))
    results=[];cache_hashes={}
    with torch.no_grad():
        for item in selected:
            start=item['start_frame'];path=args.run_dir/'gops'/f'{start:06d}.npy'
            cache_hashes[str(start)]=utils.sha256(path)
            video=torch.from_numpy(np.load(path)).float().div_(255).unsqueeze(0).to(args.device)
            valid=mask.expand(1,4,1,256,256).clone().to(args.device)
            actual=min(4,config['decoded_frames']-start);valid[:,actual:]=False
            images=[('Original',video.cpu())];baseline=None
            for mode in diagnostic.MODES:
                diagnostic.set_mode(model,mode)
                pred,_,clip=model(video)
                mse=base.masked_mse(pred,video,valid).item();psnr=-10*math.log10(max(mse,1e-12))
                if not math.isfinite(mse):raise RuntimeError('非有限MSE')
                if mode=='rounded':
                    if abs(psnr-item['psnr_db'])>.02:raise RuntimeError(f'第{start}帧基线差异超过0.02dB：当前={psnr}，记录={item["psnr_db"]}；请核对代码/设备差异')
                    baseline=psnr
                results.append(dict(start_frame=start,group=item['group'],mode=mode,mse=mse,psnr_db=psnr,
                    recorded_baseline_psnr_db=item['psnr_db'],
                    delta_psnr_db=psnr-baseline,passes=psnr>=args.threshold,
                    common_outside_original_range=clip['common'],individual_outside_original_range=clip['individual']))
                images.append((mode,pred.cpu()))
                print(f'{item["group"]} frame={start} {mode}: {psnr:.3f} dB ({psnr-baseline:+.3f})',flush=True)
            # 各组仅保存第一个固定样本，控制下载体积。
            if start==next(x['start_frame'] for x in selected if x['group']==item['group']):
                coords=mask[0,0,0].nonzero();top,left=coords.min(0).values.tolist();bottom,right=(coords.max(0).values+1).tolist()
                fig,axes=base.plt.subplots(4,4,figsize=(10,10))
                for r,(label,tensor) in enumerate(images):
                    for t in range(4):
                        axes[r,t].imshow(tensor[0,t,:,top:bottom,left:right].permute(1,2,0).numpy().clip(0,1))
                        axes[r,t].set_title(f'{label} / frame {start+t}',fontsize=9);axes[r,t].axis('off')
                fig.tight_layout();fig.savefig(out/f'{item["group"]}_frame_{start}.png',dpi=120);base.plt.close(fig)
    model.common_quantizer,model.individual_quantizer=originals
    assert all(torch.equal(v.cpu(),original_state[k]) for k,v in model.state_dict().items())
    assert utils.sha256(checkpoint)==checkpoint_hash
    summaries=[]
    for mode in diagnostic.MODES:
        r=[x for x in results if x['mode']==mode]
        summaries.append(dict(mode=mode,samples=len(r),mean_psnr_db=statistics.mean(x['psnr_db'] for x in r),
            mean_gain_db=statistics.mean(x['delta_psnr_db'] for x in r),passed=sum(x['passes'] for x in r)))
    utils.write_json(out/'result.json',dict(status='completed',epoch=epoch,optimizer_updates=0,
        checkpoint_unchanged=True,weights_unchanged=True,summary=summaries,per_gop=results,cache_sha256=cache_hashes))
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    (out/'说明.txt').write_text('从当前权重对应轮次的未达标片段中确定性选样，均值不代表全视频。\n'
        'rounded=裁剪舍入；clip_only=仅裁剪；float_identity=不裁剪不舍入。后两者非同码率条件。\n'
        '模型固定，分布适应影响直通结果；不能等价于重新训练消融，也不能据此证明量化无损。\n'
        '超范围比例始终相对于原[-1,1]，直通模式未实际裁剪。图片各组仅保存一个代表。\n',encoding='utf-8')
    bundle=out.parent/(out.name+'_brief.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in out.iterdir():archive.write(file,arcname=out.name+'/'+file.name)
    print(json.dumps(summaries,ensure_ascii=False,indent=2));print(f'诊断目录：{out}');print(f'小体积结果包：{bundle}')


if __name__=='__main__':main()
