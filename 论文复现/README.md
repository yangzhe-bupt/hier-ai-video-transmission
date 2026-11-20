# Conquering 论文复现

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
