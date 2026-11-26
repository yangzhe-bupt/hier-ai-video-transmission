"""图7激活位置独立对照；复用46的相同数据/裁剪/更新预算，默认不训练。
旧位置与相加后激活各自从头初始化。不修改41/46，不加载旧权重。
"""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'step')]
candidate = importlib.import_module('48_common_postadd_variant')
visual = importlib.import_module('53_temporal_float_model')
source = ROOT/'tools/46_reds_multivideo_ablation.py'
text = source.read_text(encoding='utf-8-sig')


def replace(old, new):
    global text
    if text.count(old) != 1:
        raise RuntimeError('46源码结构变化，停止')
    text = text.replace(old,new,1)


replace('复用41依次训练三组；不带--train仅检查。默认各30轮，全量REDS。',
        '公共分支GELU位置两组对照；不带--train仅检查。默认各30轮，队友任务显式指定60轮。')
replace("NAMES = ('spatial_only', 'residual_branches_moe_float', 'residual_branches_moe_3bit')",
        "NAMES = ('residual_branches_moe_3bit', 'residual_branches_moe_3bit_postadd')")
replace('    return codec40.factory(name)', '    return candidate_factory(name)')
replace("'outputs/step46_reds_multivideo_ablation'", "'outputs/step50_common_postadd_comparison'")
replace("name.endswith('3bit')", "(name.endswith('3bit') or name.endswith('3bit_postadd'))")
replace("paths += [ROOT/'tools/46_reds_multivideo_ablation.py']",
        "paths += [ROOT/'tools/46_reds_multivideo_ablation.py', ROOT/'tools/48_common_postadd_variant.py', ROOT/'tools/50_train_common_postadd_comparison.py', ROOT/'tools/53_temporal_float_model.py']")
replace("comparison_reference='same-data same-updates three-model ablation'",
        "comparison_reference='same-data same-initial-weights common-activation-placement ablation'")
replace("    parser.add_argument('--only', choices=NAMES)",
        "    parser.add_argument('--only', choices=NAMES)\n    parser.add_argument('--resume', type=Path)")
replace("    names = (args.only,) if args.only else NAMES", """    replace('def plot_history(out, history):', 'def legacy_plot_history(out, history):')
    replace("    write_json(out / 'initial.json', initial)", "    write_json(out / 'initial.json', initial)\\n    render_preview(model, val_data, device, out, start_epoch)")
    replace("            row['val_rounded'] = evaluate(model, val_loader, device)", "            row['val_rounded'] = evaluate(model, val_loader, device)\\n            render_preview(model, val_data, device, out, epoch)")
    replace("        write_json(out / 'history.json', history)", "        write_json(out / 'history.json', history)\\n        if epoch in (15, 30, 60):\\n            shutil.copy2(out / 'last.pt', out / f'epoch_{epoch:04d}.pt')")
    names = (args.only,) if args.only else NAMES
    resume_variant = None
    if args.resume:
        if not args.train:
            parser.error('resume需同时指定--train')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        resume_variant = saved['config']['model_variant']
        if resume_variant not in NAMES or saved['epoch'] > args.epochs:
            parser.error('续训模型或目标轮数不一致')
        if args.only and args.only != resume_variant:
            parser.error('only和续训模型不一致')
        if not args.only:
            names = NAMES[NAMES.index(resume_variant) + int(saved['epoch'] == args.epochs):]
        elif saved['epoch'] == args.epochs:
            names = ()
        del saved""")
replace("        if args.train:\n            sys.argv.append('--train')",
        "        if name == resume_variant:\n            sys.argv += ['--resume', str(args.resume)]\n        if args.train:\n            sys.argv.append('--train')")
replace("        print(f'===== {name}：目标{args.epochs}轮；从头初始化 =====', flush=True)",
        "        print(f'===== {name}：目标{args.epochs}轮；' + ('恢复Adam/RNG' if name == resume_variant else '从头初始化') + ' =====', flush=True)")
replace('codec_adapter=SimpleNamespace(factory=build), codec40=codec40,',
        'codec_adapter=SimpleNamespace(factory=build), codec40=codec40, plot_history=visual_plot, render_preview=visual_preview,')
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),candidate_factory=candidate.build_model,
     visual_plot=visual.plot_history,visual_preview=visual.preview))
