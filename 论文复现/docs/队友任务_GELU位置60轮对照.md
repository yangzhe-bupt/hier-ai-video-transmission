# 队友任务：公共分支GELU位置的60轮对照

## 目标和范围

验证图7公共特征分支最后一个GELU的位置是否影响收敛和重建。请在同一机器、同一环境依次训练两组，各60轮，从头初始化，不加载已有权重。

- A：`residual_branches_moe_3bit`，旧位置：`S + GELU(R(S))`。
- B：`residual_branches_moe_3bit_postadd`，相加后激活：`GELU(S + R(S))`。
- 保留残差组、双分支、MoE、3-bit量化，不同时改其他结构。

只移动GELU，参数名和形状不变。原位/新位置的完整模型初始权重在相同种子下已做一致性检查。新位置来自图示核对，不预先保证性能提升。

## 环境与数据

推荐Python 3.11，先安装匹配GPU的PyTorch，再安装`requirements.txt`。先记录GPU、显存、Python/PyTorch/CUDA/OpenCV版本；两个模型使用同一环境。

[REDS官方下载与校验](REDS数据集.md)：仅下载`train_sharp.zip`与`val_sharp.zip`，放在`论文复现/data/REDS/`，无需解压。数据、权重和输出均不上传Git。

在仓库的`论文复现`目录执行：

```bash
python -m pip install -r requirements.txt
python -u -B tools/48_common_postadd_variant.py
python -u -B tools/50_train_common_postadd_comparison.py --data-root data/REDS
python -u -B tools/50_train_common_postadd_comparison.py --train --epochs 60 --data-root data/REDS
```

第一条48入口仅CPU检查，第二条50入口仅检查数据；最后一条才正式训练，自动依次运行两组。保持终端/机器运行，平台不要自动关机或休眠。

## 固定实验设置

seed42、全量REDS原始训练/验证划分、GOP4、256裁剪、微批量4、累积8次、有效batch32、Adam 1e-4、MSE+0.01×MoE辅助项、FP32、无丢包。
两组共享相同洗牌和逐轮裁剪规则，裁剪随机数与量化噪声隔离。不要自行改学习率、损失、裁剪、更新量或模型大小。显存不足时先反馈，不直接改batch后混比。
每轮6000训练GOP、750验证GOP，188次参数更新；各60轮为11280次更新。

## 可视化与阶段检查点

结果根目录：`outputs/step50_common_postadd_comparison/<模型名>/<运行编号>/`。

- 每2000个GOP输出进度，每轮验证PSNR。
- `curves.png`：训练重建MSE、验证MSE/PSNR和MoE辅助损失。
- `reconstructions/epoch_XXXX.png`：同一验证GOP原图/重建/固定范围误差热图，含第0轮。
- `epoch_0015.pt`、`epoch_0030.pt`、`epoch_0060.pt`：阶段快照。
- `last.pt`、`best.pt`：最近完整轮次与本运行最佳验证模型。
- `config.json`、`data_manifest.json`、`source/`、`history.json`、`result.json`：可复查记录。

可另开终端监视曲线（独立CPU绘图，不参与训练）：

```bash
python -B tools/51_visualize_training.py --root outputs/step50_common_postadd_comparison --output outputs/training_visualization/step50 --watch
```

用浏览器打开`outputs/training_visualization/step50/dashboard.html`；页面每60秒刷新，监视器需保持运行才能更新文件。

## 中断与恢复

保持数据路径、源码与关键配置不变，从该任务自己的最新`last.pt`恢复；不能使用Step40/41/46的权重假续训。

PowerShell查找本任务最新检查点并续训：

```powershell
$last = Get-ChildItem 'outputs/step50_common_postadd_comparison/*/*/last.pt' -File |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (!$last) { throw '没有找到本任务的last.pt。' }
python -u -B tools/50_train_common_postadd_comparison.py --train --epochs 60 --data-root data/REDS --resume $last.FullName
```

恢复A组时完成后自动继续B组；恢复B组不会重跑A。保存发生在完整轮次结束，未保存的半轮会重算。严格源码校验失败时先反馈，不要删除校验或重置优化器。

## 返回结果

请先反馈第15轮，再反馈30和60轮：两组同轮次验证PSNR/MSE、训练MSE、耗时、曲线与固定场景重建图；同时记录最佳轮次，不能拿A的15轮和B的60轮比较。
完成后提供环境信息、完整终端日志、各组config/history/result、curves、重建图、阶段权重与best.pt。权重通过私下文件传输，不提交GitHub。
无需回传数据集，也不用把ZIP或视频上传仓库。
不设置必须提升的dB门槛；无增益或退化同样是有效结果。当前只验证实现位置，不做论文全部复现、实际码率或抗丢包性能宣称。
