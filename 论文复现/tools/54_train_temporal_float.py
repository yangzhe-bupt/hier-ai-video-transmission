"""不分支时空浮点多视频对照；15轮、全量REDS，默认只检查。每轮曲线与固定重建。"""
from pathlib import Path
import importlib
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'step')]
candidate = importlib.import_module('53_temporal_float_model')
source = ROOT/'tools/46_reds_multivideo_ablation.py'
text = source.read_text(encoding='utf-8-sig')


def replace(old,new):
    global text
    if text.count(old) != 1: raise RuntimeError('原入口结构变化，停止')
    text = text.replace(old,new,1)


replace("NAMES = ('spatial_only', 'residual_branches_moe_float', 'residual_branches_moe_3bit')", "NAMES = ('residual_temporal_moe_float',)")
replace('    return codec40.factory(name)', '    return candidate_factory(name)')
replace('default=30)', 'default=15)')
replace("'outputs/step46_reds_multivideo_ablation'", "'outputs/step54_temporal_float'")
replace("paths += [ROOT/'tools/46_reds_multivideo_ablation.py']", "paths += [ROOT/'tools/46_reds_multivideo_ablation.py', ROOT/'tools/53_temporal_float_model.py', ROOT/'tools/54_train_temporal_float.py']")
replace("cbr_dimension_ratio=(.375 if args.model_variant == 'spatial_only' else 16896 / (4 * 3 * 256 * 256))",
        'cbr_dimension_ratio=30720 / (4 * 3 * 256 * 256)')
replace("comparison_reference='same-data same-updates three-model ablation'", "comparison_reference='step46; no-branch temporal-float; different latent budgets'")
replace("    names = (args.only,) if args.only else NAMES", """    replace('def plot_history(out, history):', 'def legacy_plot_history(out, history):')
    replace("    write_json(out / 'initial.json', initial)", "    write_json(out / 'initial.json', initial)\\n    render_preview(model, val_data, device, out, start_epoch)")
    replace("            row['val_rounded'] = evaluate(model, val_loader, device)", "            row['val_rounded'] = evaluate(model, val_loader, device)\\n            render_preview(model, val_data, device, out, epoch)")
    names = (args.only,) if args.only else NAMES""")
replace('codec_adapter=SimpleNamespace(factory=build), codec40=codec40,',
        'codec_adapter=SimpleNamespace(factory=build), codec40=codec40, plot_history=visual_plot, render_preview=visual_preview,')
exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),
    candidate_factory=candidate.build,visual_plot=candidate.plot_history,visual_preview=candidate.preview))
