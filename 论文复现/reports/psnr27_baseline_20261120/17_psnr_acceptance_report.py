"""从既有训练记录统计逐GOP画质验收，不更新模型或修改源文件。

27 dB为用户针对当前动画内容确认的暂定标准；不是通用视觉阈值。
帧覆盖率表示所在GOP达标的帧比例，并非每帧独立PSNR验收。
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=27.)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    if not math.isfinite(args.threshold):p.error('阈值必须有限')
    config=json.loads((args.run_dir/'config.json').read_text(encoding='utf-8'))
    result=json.loads((args.run_dir/'result.json').read_text(encoding='utf-8'))
    count=config['decoded_frames'];fps=config['fps']
    with (args.run_dir/'per_gop.csv').open(encoding='utf-8') as f:raw=list(csv.DictReader(f))
    by_epoch={}
    for r in raw:
        epoch=int(r['epoch']);start=int(r['start_frame']);mse=float(r['eval_mse']);psnr=float(r['eval_psnr_db'])
        assert abs(-10*math.log10(mse)-psnr)<1e-6
        by_epoch.setdefault(epoch,[]).append(dict(start_frame=start,stop_frame=min(start+4,count),mse=mse,
            psnr_db=psnr,passes=psnr>=args.threshold,detail_psnr_db=float(r['detail_psnr_db'])))
    summaries=[]
    for epoch,rows in sorted(by_epoch.items()):
        assert len(rows)==len(config['gop_starts']) and {r['start_frame'] for r in rows}==set(config['gop_starts'])
        frames=sum(r['stop_frame']-r['start_frame'] for r in rows)
        accepted=[r for r in rows if r['passes']]
        mse=sum(r['mse']*(r['stop_frame']-r['start_frame']) for r in rows)/frames
        scores=sorted(r['psnr_db'] for r in rows)
        summaries.append(dict(epoch=epoch,gops=len(rows),passed_gops=len(accepted),failed_gops=len(rows)-len(accepted),
            gop_pass_rate=len(accepted)/len(rows),frame_coverage_rate=sum(r['stop_frame']-r['start_frame'] for r in accepted)/frames,
            pooled_mse=mse,pooled_psnr_db=-10*math.log10(mse),minimum_psnr_db=scores[0],
            p10_psnr_db=scores[int((len(scores)-1)*.1)],median_psnr_db=statistics.median(scores),maximum_psnr_db=scores[-1]))
    final_epoch=int(result['final']['epoch'])
    latest=next(r for r in summaries if r['epoch']==final_epoch)
    assert abs(latest['pooled_mse']-result['final']['mean_mse'])<1e-8
    rows=sorted(by_epoch[final_epoch],key=lambda r:r['start_frame'])
    # 连续未通过区间，不把隔开的失效位置合并。
    intervals=[]
    for r in rows:
        if r['passes']:continue
        if intervals and intervals[-1]['stop_frame']==r['start_frame']:
            intervals[-1]['stop_frame']=r['stop_frame'];intervals[-1]['gops']+=1
        else:intervals.append(dict(start_frame=r['start_frame'],stop_frame=r['stop_frame'],gops=1))
    for interval in intervals:
        interval.update(start_seconds=interval['start_frame']/fps,stop_seconds=interval['stop_frame']/fps)
    worst=sorted(rows,key=lambda r:r['psnr_db'])[:20]
    for r in worst:r.update(start_seconds=r['start_frame']/fps,stop_seconds=r['stop_frame']/fps)
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    report=dict(threshold_db=args.threshold,threshold_source='user acceptance of fixed animation GOP',
        source_run=str(args.run_dir),source_sha256=hashlib.sha256((args.run_dir/'per_gop.csv').read_bytes()).hexdigest(),
        final=latest,epoch_history=summaries,worst20=worst,failed_intervals=intervals,
        interpretation='frame coverage uses GOP pass/fail; times use mean FPS; this is training-set quality, not generalization',
        optimizer_updates=0)
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(args.run_dir/'per_gop.csv',out/'source_per_gop.csv')
    shutil.copy2(Path(__file__),out/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist([r['psnr_db'] for r in rows],bins=30,color='#337bab',edgecolor='white')
    axes[0].axvline(args.threshold,color='#b23c36',linestyle='--',label=f'Threshold {args.threshold:g} dB')
    axes[0].set(xlabel='Per-GOP PSNR (dB)',ylabel='GOP count');axes[0].legend()
    axes[1].plot([r['start_frame']/fps for r in rows],[r['psnr_db'] for r in rows],linewidth=.7)
    axes[1].axhline(args.threshold,color='#b23c36',linestyle='--')
    axes[1].set(xlabel='Time from frame index / mean FPS (s)',ylabel='Per-GOP PSNR (dB)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f'Epoch {final_epoch} / {latest["passed_gops"]}/{latest["gops"]} GOPs pass / training video')
    fig.tight_layout();fig.savefig(out/'acceptance_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([r['epoch'] for r in summaries],[r['gop_pass_rate']*100 for r in summaries],'o-')
    ax.set(xlabel='Epoch',ylabel=f'GOPs at least {args.threshold:g} dB (%)',ylim=(0,100));ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'pass_rate_history.png',dpi=160);plt.close(fig)
    lines=[f'画质验收基线：第{final_epoch}轮，暂定阈值{args.threshold:g} dB。',
        f'达标GOP：{latest["passed_gops"]}/{latest["gops"]}（{latest["gop_pass_rate"]:.2%}）。',
        f'未达标GOP：{latest["failed_gops"]}；所在GOP达标的帧覆盖率：{latest["frame_coverage_rate"]:.2%}。',
        f'汇总PSNR：{latest["pooled_psnr_db"]:.3f} dB；最低：{latest["minimum_psnr_db"]:.3f}；中位：{latest["median_psnr_db"]:.3f}；最高：{latest["maximum_psnr_db"]:.3f}。',
        '以下时间由帧号/平均帧率计算，非逐帧原始时间戳。帧区间左闭右开。',
        '最差20段（起始帧 / 秒 / PSNR）：']
    lines += [f'{r["start_frame"]} / {r["start_seconds"]:.3f}s / {r["psnr_db"]:.3f} dB' for r in worst]
    lines += ['达标仅表示超过当前内容的数值验收线，仍需检查模糊、闪烁及块状伪影。',
              '阈值来自用户对单个动画GOP的画质偏好，不是通用标准；没有重新训练。',
              '后续比较应同时观察汇总、达标率、最差片段与视觉效果。']
    (out/'验收基线.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(latest,ensure_ascii=False,indent=2))
    print('最差5段：',[(r['start_frame'],round(r['start_seconds'],2),round(r['psnr_db'],3)) for r in worst[:5]])
    print(f'验收记录保存：{out}')


if __name__=='__main__':main()
