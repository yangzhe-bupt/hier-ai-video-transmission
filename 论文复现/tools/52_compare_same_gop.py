"""对相同验证GOP绘制原图/三模型重建，CPU只读。"""
import csv
import argparse
import importlib
from pathlib import Path
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'step')]
factory = importlib.import_module('48_common_postadd_variant')
compare = importlib.import_module('30_compare_reds_train_val')
torch.set_num_threads(2)
compare.data.cv2.setNumThreads(1)
torch.manual_seed(42)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation-root',type=Path,required=True,help='49号工具的三模型评估目录')
p.add_argument('--data-root',type=Path,default=Path('D:/Datasets/REDS'))
p.add_argument('--output',type=Path,default=ROOT/'outputs/training_visualization/step46')
args = p.parse_args()
base = args.evaluation_root
rows = list(csv.DictReader((base/'02_residual_branches_moe_3bit/per_gop.csv').open(encoding='utf-8-sig')))
ranked = sorted(rows,key=lambda r:float(r['psnr_db']))
samples = [('worst',ranked[0]),('median',ranked[len(ranked)//2])]
ds = compare.CenterDataset(args.data_root/'val_sharp.zip','val')
models = []
for sub in ('00_spatial_only','01_residual_branches_moe_float','02_residual_branches_moe_3bit'):
    import json
    info = json.loads((base/sub/'result.json').read_text(encoding='utf-8'))
    saved = torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
    m = factory.build_model(info['variant']).eval().requires_grad_(False)
    m.load_state_dict(saved['model'],strict=True)
    models.append((info['variant'],m))
out = args.output
out.mkdir(parents=True,exist_ok=True)
with torch.no_grad():
    for label,row in samples:
        x = ds[int(row['dataset_index'])]['video'].unsqueeze(0)
        images = [('Original',x[0])]
        for name,m in models:
            y = m(x)[0]
            mse = (y-x).square().double().mean().item()
            images.append((f'{name} {compare.diag.psnr(mse):.3f} dB',y[0]))
        fig,axes = plt.subplots(4,4,figsize=(12,12))
        for i,(name,frames) in enumerate(images):
            for t in range(4):
                axes[i,t].imshow(frames[t].permute(1,2,0).clamp(0,1).numpy())
                axes[i,t].set_title(f'Frame {int(row["start"])+t}',fontsize=9)
                axes[i,t].axis('off')
            axes[i,0].text(0,1.15,name,transform=axes[i,0].transAxes,fontsize=9)
        fig.suptitle(f'Same GOP comparison: {row["sequence"]}, selected by quantized {label}')
        fig.tight_layout(); fig.savefig(out/f'same_gop_{label}.png',dpi=130);plt.close(fig)
        print(out/f'same_gop_{label}.png',flush=True)
ds.close()
