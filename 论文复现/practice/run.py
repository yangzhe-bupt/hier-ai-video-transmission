# ======================================================================
# 文件：run.py
# 功能：在不修改原脚本导入和路径计算的条件下，运行分类后的练习文件。
# 类与函数汇总：
#   - LegacyLoader：读取真实文件，提供原practice目录下的逻辑文件路径。
#   - PracticeFinder：按原模块名称定位各子文件夹中的文件。
#   - main：列出、检查导入或运行选定脚本。
# 用法：python D:/learn/pytorch/practice/run.py --list
#       python D:/learn/pytorch/practice/run.py models/swin/swin_block_pair.py
# 注意：原脚本依赖统一入口；__file__使用移动前路径以兼容项目定位。
# ======================================================================
import argparse
import importlib
import importlib.abc
import importlib.util
import linecache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FILES = {p.stem: p for p in ROOT.rglob('*.py')
         if p.name != 'run.py' and '__pycache__' not in p.parts}


class LegacyLoader(importlib.abc.Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        virtual = str(ROOT / self.path.name)
        source = self.path.read_text(encoding='utf-8-sig')
        module.__file__ = virtual
        linecache.cache[virtual] = (len(source), None, source.splitlines(True), virtual)
        exec(compile(source, virtual, 'exec'), module.__dict__)


class PracticeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if '.' not in fullname and fullname in FILES:
            return importlib.util.spec_from_loader(
                fullname, LegacyLoader(FILES[fullname]),
                origin=str(ROOT / FILES[fullname].name))
        return None


def main():
    parser = argparse.ArgumentParser(description='分类练习文件统一入口')
    parser.add_argument('script', nargs='?')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--verify-imports', action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, PracticeFinder())
    sys.path.insert(0, str(ROOT.parent / 'Biran'))
    if args.list:
        for p in sorted(FILES.values()):
            print(p.relative_to(ROOT))
        return
    if args.verify_imports:
        for name, p in sorted(FILES.items()):
            if p.parent.name == 'basics':
                continue
            module = importlib.import_module(name)
            assert Path(module.__file__).parent == ROOT
            print('导入与逻辑路径通过：', name)
        return
    if not args.script:
        parser.error('请提供脚本路径或使用--list')
    candidate = (ROOT / args.script).resolve()
    if candidate not in FILES.values():
        candidate = FILES.get(Path(args.script).stem)
    if candidate is None or candidate not in FILES.values():
        parser.error('脚本不在已整理的practice文件中')
    import types
    module = types.ModuleType('__main__')
    module.__package__ = None
    sys.modules['__main__'] = module
    sys.argv = [str(candidate)]
    LegacyLoader(candidate).exec_module(module)


if __name__ == '__main__':
    main()
