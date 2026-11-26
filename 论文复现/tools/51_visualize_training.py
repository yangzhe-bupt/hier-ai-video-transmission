"""只读训练可视化；可用--watch监视新增轮次，不修改训练脚本。"""
import argparse
import base64
from datetime import datetime
import html
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def render(root, out):
    selected = {}
    for path in root.glob('*/*/history.json'):
        try:
            rows = json.loads(path.read_text(encoding='utf-8-sig'))
            if not rows:
                continue
            name = path.parent.parent.name
            rank = (rows[-1]['epoch'],path.stat().st_mtime_ns)
            if name not in selected or rank > selected[name][0]:
                selected[name] = (rank,path,rows)
        except (OSError,ValueError,KeyError):
            continue
    if not selected:
        raise RuntimeError('尚无可视化历史记录')
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    summaries = []
    for name, (_,path,rows) in sorted(selected.items()):
        e = [r['epoch'] for r in rows]
        v = [r for r in rows if 'val_rounded' in r]
        axes[0,0].plot([r['epoch'] for r in v],[r['val_rounded']['psnr_db'] for r in v],label=name)
        axes[0,1].plot(e,[r['train_noise_mse'] for r in rows],label=name)
        axes[1,0].plot([r['epoch'] for r in v],[r['val_rounded']['mse'] for r in v],label=name)
        axes[1,1].plot(e,[r['train_aux'] for r in rows],label=name)
        best = max(v,key=lambda r:r['val_rounded']['psnr_db']) if v else None
        config = json.loads((path.parent/'config.json').read_text(encoding='utf-8-sig'))
        summaries.append(dict(variant=name,run=str(path.parent),epoch=e[-1],target=config['epochs'],
                              updates=rows[-1]['updates'],best_epoch=best['epoch'] if best else None,
                              best_val_psnr=best['val_rounded']['psnr_db'] if best else None))
    for ax, title, ylabel in zip(axes.flat,('Validation PSNR','Train reconstruction MSE','Validation MSE','MoE auxiliary'),
                                 ('PSNR (dB)','MSE','MSE','Auxiliary')):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel)
        ax.grid(alpha=.3)
        if 'MSE' in title:
            ax.set_yscale('log')
    axes[0,0].legend(fontsize=7)
    fig.tight_layout()
    png = out/'curves.png'
    fig.savefig(png,dpi=150)
    plt.close(fig)
    def cell(x): return html.escape(str(x))
    table = ''.join('<tr>'+''.join(f'<td>{cell(r[k])}</td>' for k in
                    ('variant','epoch','target','updates','best_epoch','best_val_psnr'))+'</tr>' for r in summaries)
    picture = base64.b64encode(png.read_bytes()).decode()
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<title>训练可视化</title><style>body{{font:16px sans-serif;max-width:1250px;margin:30px auto;padding:16px}}td,th{{border:1px solid #ccc;padding:8px}}table{{border-collapse:collapse}}img{{width:100%}}</style>
<h1>训练可视化</h1><p>生成：{datetime.now():%Y-%m-%d %H:%M:%S}。只读历史，不更新模型。</p>
<p>下表按各变体最高保存轮次展示。比较时应选相同轮次/更新量；空间与完整模型不是等码率对照。</p>
<table><tr><th>模型</th><th>已保存轮数</th><th>目标轮数</th><th>参数更新</th><th>最佳轮次</th><th>最佳验证PSNR</th></tr>{table}</table>
<img src="data:image/png;base64,{picture}" alt="训练与验证曲线">
<p>训练MSE来自实际训练路径：量化组含噪声、浮点组不含噪声，不能当成同口径推理误差。推荐结合统一验证与同场景重建图。</p>
<p>本页每60秒重载；只有--watch运行时才会更新页面文件。默认CPU绘图，不占训练GPU。</p></html>'''
    temp = out/'dashboard.tmp.html'
    temp.write_text(body,encoding='utf-8'); temp.replace(out/'dashboard.html')
    (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'可视化已更新：{out}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'outputs/step46_reds_multivideo_ablation')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/training_visualization/step46')
    p.add_argument('--watch',action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    while True:
        render(args.root,args.output)
        if not args.watch: break
        time.sleep(60)


if __name__ == '__main__': main()
