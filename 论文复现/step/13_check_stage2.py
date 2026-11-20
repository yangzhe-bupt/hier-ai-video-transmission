"""阶段II联合检查，不创建优化器、不更新权重。"""
import argparse
from datetime import datetime
import importlib
from pathlib import Path
import json
import torch

s2 = importlib.import_module('13_stage2_model')
data = importlib.import_module('12_prepare_multi_gop')
utils = importlib.import_module('12_train_single_gop')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent.parent/'outputs/step13_stage2_checks')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    dataset = data.CachedGOPDataset(args.data_dir,'val')
    if utils.sha256(args.data_dir/'manifest.json') != saved['config']['manifest_sha256']:
        raise ValueError('数据清单不同')
    if utils.sha256(args.data_dir/'frames_uint8.npy') != dataset.manifest['cache_sha256']:
        raise ValueError('缓存哈希不同')
    codec = s2.variants.build_model(saved['config'].get('model_variant','legacy'))
    codec.load_state_dict(saved['model'])
    codec.to(args.device).eval()
    sample = dataset[0]
    video = sample['video'].unsqueeze(0).to(args.device)
    valid = sample['valid_mask'].unsqueeze(0).to(args.device)
    with torch.no_grad(): reference = codec(video)[0]
    model = s2.Stage2Model(codec).to(args.device)
    original = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.train()
    with torch.no_grad():
        no_loss,_,feature,_ = model(video,[0.],[42])
        torch.testing.assert_close(no_loss,reference,rtol=0,atol=0)
        assert feature.item()==0
        # mask逆交织之后，缺失处应置零，收到处应逐位相同。
        model(video,[1.],[42])
        assert all(r['missing_mask'].all() and (r['received']==0).all() for r in model.branches)
    prediction,aux,features,_ = model(video,[.5],[42])
    reconstruction = s2.base.masked_mse(prediction,video,valid)
    # 仅供检查的系数；不是论文确认的alpha/beta/gamma。
    loss = reconstruction+.01*aux+.1*features
    loss.backward()
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
        elif parameter.grad is None:
            assert '.experts.' in name, name
        else:
            assert torch.isfinite(parameter.grad).all(),name
    # 除总损失以外，视频重建损失本身也应连通双恢复网络。
    model.zero_grad(set_to_none=True)
    prediction,_,_,_ = model(video,[.5],[42])
    s2.base.masked_mse(prediction,video,valid).backward()
    for module in (model.common_recovery,model.individual_recovery):
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in module.parameters())
        assert sum(q.grad.abs().sum().item() for q in module.parameters())>0
    assert all(torch.equal(v.cpu(),original[k]) for k,v in model.state_dict().items())
    report = dict(device=args.device,torch=torch.__version__,optimizer_updates=0,
        checkpoint=str(args.checkpoint),variant=saved['config'].get('model_variant','legacy'),
        zero_loss_matches_stage1=True, observed_elements_unchanged=True,full_loss_mask_checked=True,
        frozen_parameters_have_no_grad=True,reconstruction_reaches_both_recovery_networks=True,
        weights_unchanged=True, frozen_modules=list(model.frozen_names),
        trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),
        reconstruction_mse_at_50pct=reconstruction.item(),feature_loss=features.item(),
        feature_target='rounded transmitted features (implementation choice)',
        quantization='frozen sender eval rounding',feature_loss_reduction='sum of two whole-branch means')
    out=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    utils.write_json(out/'checks.json',report)
    import shutil
    for file in (Path(__file__),Path(s2.__file__)):
        shutil.copy2(file,out/file.name)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'检查记录：{out}')


if __name__=='__main__': main()
