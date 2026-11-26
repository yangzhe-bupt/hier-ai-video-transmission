# Conquering 论文复现

> **2026-11-26 更新：** 新增多视频三组对照与恢复（46/47）、图7激活位置变体（48/50）、统一评估（49）、曲线面板/同场景对比（51/52）、不分支时空浮点模型（53/54）。队友长任务见[60轮GELU位置对照任务](docs/队友任务_GELU位置60轮对照.md)，包含每轮重建、曲线、阶段快照和恢复命令。只提供[数据集下载地址](docs/REDS数据集.md)，不上传数据与权重。

> **2026-11-25 当前主线更新：** 残差组时空编解码 + 双分支 + MoE + 3-bit量化，已进入REDS多视频Stage I训练。请优先阅读[当前训练入口](docs/当前训练入口.md)、[模型交接说明](docs/模型代码交接说明_20261124.txt)和[数据集下载](docs/REDS数据集.md)。下文为历史单视频与Stage II入口，当前残差多视频模型尚未完成有损信道与工程指标验收。数据、权重及本次训练输出不上传。

基于 Teng 等的 *Conquering High Packet-Loss Erasure: MoE Swin Transformer-Based Video Semantic Communication* 搭建 MSTVSC 学习原型。已完成无丢包重建训练、模拟整包擦除下的抗丢包接收端训练，以及整段视频重建和导出。当前用 common_fig7 连接阶段 II。真实字节包与自适应反馈是后续集成项。

## 目录

| 路径 | 用途 |
| --- | --- |
| [step/](step/) | 按阶段组织复现脚本。12 系列为阶段 I，13 系列为丢包和阶段 II，14–17 为诊断、拟合、导出和验收报告。 |
| [practice/](practice/)、[Biran/](Biran/)、[packet/](packet/) | 复现脚本沿用的模型、GOP 数据读取、特征分解和丢包工具。 |
| [references/](references/) | 论文结构核对，以及早期分包和 UDP 代码的参考副本。 |
| [docs/](docs/) | 复现计划、历史交接与来源清单。 |
| [reports/](reports/) | 已有图表、汇报和轻量运行记录。 |
| [smoke_baseline.py](smoke_baseline.py) | 随机张量检查，无需视频和检查点。 |
| data/、outputs/ | 本机的视频、预处理数据、权重和生成产物。详见[本地资产清单](docs/本地资产清单.csv)。 |

## 快速开始

在本目录运行以下命令。使用 Python 3.10 或 3.11 和匹配本机的 PyTorch。GPU 环境请先安装对应 CUDA 版本的 PyTorch。

~~~powershell
python -m pip install -r requirements.txt
python -B smoke_baseline.py
python -B step/13_check_stage2.py --help
~~~

有输入视频后，可以按需准备 GOP、训练和评估。将下面的变量替换为实际的输入、准备脚本输出目录和模型检查点路径。

~~~powershell
python -B step/12_prepare_multi_gop.py --video $video
python -B step/12_train_multi_gop.py --data-dir $gopData --model-variant common_fig7 --device cuda
python -B step/13_train_stage2.py --data-dir $gopData --checkpoint $stage1 --device cuda
python -B step/13_evaluate_stage2.py --data-dir $gopData --checkpoint $stage2 --device cuda
python -B step/16_reconstruct_video.py --video $video --checkpoint $fitCheckpoint --device cuda --max-gops 4
~~~

训练和验证示例使用同一视频的时间段。报告指标应注明实际划分和压缩预算。现有结果见[阶段汇报](reports/20261120/阶段汇报与结果说明.txt)。集成自适应时，保留同一编解码权重下的固定发包策略。

## 本地资产

将自备视频与预处理数据放在 data/，将对应检查点和完整实验产物放在 outputs/。上传目录保留按运行编号归档的配置、指标和记录，原始资源位置与大小见 [本地资产清单](docs/本地资产清单.csv)。部分历史脚本仍引用原作者机器的素材路径，运行前以实际路径和 CLI 参数为准。
