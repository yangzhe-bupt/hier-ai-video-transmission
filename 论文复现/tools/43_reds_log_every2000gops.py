"""复用41；每轮每处理2000个GOP输出一次，不改变训练或源码哈希。"""
from pathlib import Path

SOURCE = Path(__file__).resolve().with_name('41_train_reds_residual.py')
source = SOURCE.read_text(encoding='utf-8-sig')
old = '            if (index + 1) % 25 == 0 or index + 1 == len(loader):'
if source.count(old) != 1:
    raise RuntimeError('训练脚本日志结构变化，停止；未修改原文件')
source = source.replace(old, '            if seen % 2000 == 0 or index + 1 == len(loader):', 1)
exec(compile(source, str(SOURCE), 'exec'), {'__name__': '__main__', '__file__': str(SOURCE)})
