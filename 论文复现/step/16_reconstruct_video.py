"""冻结模型顺序重建整段视频，导出有效区域及并排对比；不训练。

末尾不足4帧时重复最后一帧补齐，仅导出和评价实际帧。
MP4为mp4v有损展示文件，指标在视频压缩前的浮点重建上计算。
按源元数据平均fps导出恒定帧率，无音频，不恢复原始960×720细节。
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

import cv2
import numpy as np
import torch

variants=importlib.import_module('12_model_variants')
base=variants.base
utils=importlib.import_module('12_train_single_gop')
ROOT=Path(__file__).resolve().parent.parent


def verify_video(path,expected_frames,expected_size):
    """重新完整解码导出文件，检查帧数、尺寸及可读性。"""
    cap=cv2.VideoCapture(str(path));count=0
    try:
        if not cap.isOpened():raise RuntimeError(f'无法打开导出文件：{path}')
        while True:
            ok,frame=cap.read()
            if not ok:break
            if (frame.shape[1],frame.shape[0])!=expected_size:raise RuntimeError('导出尺寸不一致')
            count+=1
    finally:cap.release()
    if count!=expected_frames:raise RuntimeError(f'{path.name}帧数错误：{count} != {expected_frames}')
    return count


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--max-gops',type=int,default=0,help='0处理全部；正值只用于短检查')
    p.add_argument('--output-root',type=Path,default=ROOT/'outputs/step16_reconstructed_video')
    args=p.parse_args()
    if args.max_gops<0:p.error('max-gops必须非负')
    torch.set_num_threads(2)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']
    if utils.sha256(args.video)!=config['video_sha256']:raise ValueError('视频哈希与训练记录不同')
    model=variants.build_model(config['model_variant']).to(args.device)
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    checkpoint_hash=utils.sha256(args.checkpoint)
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened():raise RuntimeError('视频无法打开')
    fps=cap.get(cv2.CAP_PROP_FPS);expected_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps<=0:cap.release();raise RuntimeError('源fps不可用')
    height,width=config['preprocessing']['resized_hw']
    top,_,left,_=config['preprocessing']['padding_tblr']
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    paths=[out/'reconstructed.mp4',out/'comparison.mp4']
    sizes=[(width,height),(width*2,height+24)]
    writers=[cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size) for path,size in zip(paths,sizes)]
    if not all(w.isOpened() for w in writers):
        cap.release()
        for w in writers:w.release()
        raise RuntimeError('mp4v写入器无法打开')
    utils.write_json(out/'config.json',dict(checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=saved['epoch'],video=str(args.video),video_sha256=config['video_sha256'],
        fps=fps,source_metadata_frames=expected_count,output_sizes=sizes,device=args.device,
        torch=torch.__version__,max_gops=args.max_gops,tail='repeat final frame, evaluate/export actual frames only',
        optimizer_updates=0,metrics='float predictions before MP4 encoding; PSNR from pooled RGB MSE'))
    utils.write_json(out/'training_config.json',config)
    for folder in ('step','practice','Biran','packet'):
        for source in (ROOT/folder).rglob('*.py'):
            target=out/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(Path(base.step05.__file__),out/'source/step/05_video_dataset.py')
    groups={k:dict(squared_error=0.,pixels=0,frames=0,gops=0) for k in ('trained','other','all')}
    training_starts=set(config['gop_starts'])
    frame_count=0;gop_count=0;tail_count=0;rows=[];started=time.perf_counter()
    print(f'重建目录：{out}；checkpoint epoch={saved["epoch"]}；仅推理',flush=True)
    try:
        with (out/'per_gop.csv').open('w',newline='',encoding='utf-8') as stream,torch.no_grad():
            writer=csv.DictWriter(stream,fieldnames=['start_frame','actual_frames','group','mse','psnr_db']);writer.writeheader()
            while args.max_gops==0 or gop_count<args.max_gops:
                frames=[]
                for _ in range(4):
                    ok,frame=cap.read()
                    if not ok:break
                    frames.append(frame)
                if not frames:break
                actual=len(frames);start=frame_count
                if actual<4:tail_count=actual
                padded=frames+[frames[-1]]*(4-actual)
                video,mask,info=base.step05.frames_to_full_gop(padded,256)
                if json.loads(json.dumps(info))!=json.loads(json.dumps(config['preprocessing'])):raise ValueError('预处理与训练记录不同')
                prediction=model(video.to(args.device))[0].cpu()[:,:actual]
                original=video[:,:actual]
                square=torch.where(mask,(prediction-original).square(),0.).double().sum().item()
                pixels=mask.expand_as(original).sum().item()
                mse=square/pixels
                if not math.isfinite(mse):raise RuntimeError('非有限重建误差')
                group='trained' if start in training_starts else 'other'
                for name in (group,'all'):
                    g=groups[name];g['squared_error']+=square;g['pixels']+=pixels;g['frames']+=actual;g['gops']+=1
                row=dict(start_frame=start,actual_frames=actual,group=group,mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)))
                rows.append(row);writer.writerow(row);stream.flush()
                for t in range(actual):
                    rgb=(prediction[0,t,:,top:top+height,left:left+width].permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
                    ref=(original[0,t,:,top:top+height,left:left+width].permute(1,2,0)*255).round().byte().numpy()
                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);ref_bgr=cv2.cvtColor(ref,cv2.COLOR_RGB2BGR)
                    writers[0].write(bgr)
                    comparison=np.zeros((height+24,width*2,3),dtype=np.uint8)
                    comparison[24:,:width]=ref_bgr;comparison[24:,width:]=bgr
                    cv2.putText(comparison,f'Input | frame {start+t}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    cv2.putText(comparison,f'Reconstruction | {group}',(width+3,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    writers[1].write(comparison)
                frame_count+=actual;gop_count+=1
                if gop_count%50==0:print(f'已重建{frame_count}/{expected_count}帧',flush=True)
    finally:
        cap.release()
        for w in writers:w.release()
    if args.max_gops==0 and frame_count!=expected_count:raise RuntimeError('完整解码帧数与源元数据不同')
    for path,size in zip(paths,sizes):verify_video(path,frame_count,size)
    assert utils.sha256(args.checkpoint)==checkpoint_hash
    results={}
    for name,g in groups.items():
        mse=g['squared_error']/g['pixels'] if g['pixels'] else None
        results[name]=dict(frames=g['frames'],gops=g['gops'],mse=mse,psnr_db=-10*math.log10(max(mse,1e-12)) if mse is not None else None)
    best_metrics=saved['metrics']
    trained_match=None
    if args.max_gops==0:
        trained_match=abs(results['trained']['mse']-best_metrics['mean_mse'])<1e-6
        if not trained_match:raise RuntimeError('已训练GOP未复现检查点MSE，请保留记录排查')
    report=dict(status='completed',partial=args.max_gops>0,checkpoint_epoch=saved['epoch'],
        output_frames=frame_count,gops=gop_count,tail_actual_frames=tail_count,output_fps=fps,
        duration_seconds_cfr=frame_count/fps,groups=results,trained_metrics_reproduced=trained_match,
        video_readback_verified=True,checkpoint_unchanged=True,optimizer_updates=0,elapsed_seconds=time.perf_counter()-started)
    utils.write_json(out/'result.json',report)
    fig,ax=base.plt.subplots(figsize=(12,4))
    ax.plot([r['start_frame'] for r in rows],[r['psnr_db'] for r in rows],linewidth=.8,label='All GOPs')
    seen=[r for r in rows if r['group']=='trained']
    ax.scatter([r['start_frame'] for r in seen],[r['psnr_db'] for r in seen],color='red',s=24,label='Training GOPs')
    ax.set(xlabel='Start frame',ylabel='Rounded PSNR (dB)',title='Full-video reconstruction diagnostic');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'per_gop_curve.png',dpi=160);base.plt.close(fig)
    (out/'summary.txt').write_text('顺序重建单视频，不更新模型。trained为原8个GOP；other为其余GOP，不是跨视频独立测试。\n'
        '末尾不足四帧重复最后帧补齐，重复帧不导出、不参与指标。\n'
        '重建视频为256×192有效区域；对比视频左侧为同预处理输入，而非原分辨率。\n'
        'MP4有损编码仅用于观看，指标基于编码前浮点输出。无音频；按元数据平均fps导出CFR，未保留源文件逐帧时间戳。\n'
        '完整播放质量需实际查看；单帧PSNR不能证明时间一致性。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print(f'重建保存完成：{out}',flush=True)


if __name__=='__main__':main()
