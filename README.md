# PING-III 层次化AI视频传输

面向动态网络的语义视频AI层次化编解码与丢包鲁棒自适应传输研究。

## 当前进度（杨哲，第二周，2026-11-01）

已完成轻量CNN基线、二维Swin编码器与CNN解码器训练、最佳权重保存加载，以及毕然公共/个体特征模块的无丢包联调。尚未形成完整语义传输系统，也不代表GRACE或MSTVSC完整复现。

阶段目标：720P@24fps源视频、约1Mbps实际总发送码率、10%丢包、LPIPS<0.6，均待验证。当前实验为128×128中心裁剪、4帧GOP、浮点特征、MSE训练。

|成员|方向|
|---|---|
|杨哲|语义编解码主干、丢包模拟、缺失恢复和鲁棒性实验|
|毕然|特征分解、压缩、量化、分包与交织|
|蒋立果|网络条件、预算、期限、反馈与自适应；分层接口需共同确认|

## 目录

```text
practice/
  data/            # GOP索引及视频Dataset
  models/          # CNN及Swin编码器、CNN解码器
    swin/          # 窗口注意力、位置偏置、移位掩码、块对
  experiments/     # 训练、验证和图像对比
  checks/          # 编解码器与毕然模块联调
  run.py           # 兼容原导入和__file__项目定位
Biran/feature_decomposition.py # 当前联调依赖
hier_video/        # 第一周整理的基线包，仍保留
checks/            # 第一周基础检查
results/week2/     # 固定样例与历史指标
docs/datasets/     # 视频来源说明
context            # 各周成果与证据边界
```

## 环境与运行

已使用Conda、PyTorch 2.6.0+cu124、RTX4060 Laptop GPU约8GB。保留可用GPU环境；基础依赖见requirements.txt。

在仓库根目录运行：

```powershell
python practice/run.py --list
python practice/run.py models/swin/swin_block_pair.py
python practice/run.py models/swin_encoder.py
python practice/run.py checks/check_biran_integration.py
```

请通过run.py运行分类后的脚本，不直接运行子目录文件。主线训练数据放在仓库根目录下的data/videos/train和data/videos/val，测试目录单独保留。模型输出到outputs。训练和验证目前各一个来源视频，不代表丰富的多视频训练集。

```powershell
python practice/run.py experiments/train_swin_video.py
```

当前Swin训练最多500轮；连续30轮未改善超过1e-5时早停。新实验输出到outputs/swin_runs/时间戳目录：best.pt用于评测，latest.pt保存完整轮边界续训状态，history.csv记录逐轮MSE。设置脚本中的resume_run_dir可恢复下一轮；中断轮需要重做。数据、模型配置及运行环境应保持一致。

早期单GOP、CNN训练和部分联调脚本仍包含原作者D盘路径，使用前需在自己的副本调整。compare_cnn_swin.py默认读取此前独立的五轮基线文件名；若使用新实验best.pt，需调整其检查点路径。权重和原始视频未纳入Git，缺少权重时不能直接运行重建评测。Swin权重必须由load_swin_models加载，CNN的load_models不能加载它。

## 第二周实验记录

以下全验证指标来自杨哲已完成的五轮运行反馈，本次发布未重训：

|配置|最佳验证MSE|
|---|---:|
|CNN编码器＋CNN解码器|0.010229|
|Swin编码器＋CNN解码器|0.009117|

Swin在该单次实验中平均MSE较低，但两侧模型容量不同，且部分片段退化；不代表稳定的感知质量或语义收益。对照图显示两者仍有模糊、偏色。三个固定片段指标和图片见[results/week2](results/week2)。其中比较CSV的image_path为原运行机器的历史路径，对应图片也已放在同一结果目录。

毕然模块：倍数1在两组真实GOP上保持原重建，倍数2可运行但增加损失。公共/个体特征不自动等于基础/增强层。

代码接口：RGB输入[B,T,3,128,128]，编码特征[B,T,24,32,32]，重建[B,T,3,128,128]。各帧共享参数但独立处理，无跨帧建模；特征尚未量化、独立字节分包。当前丢包模块为特征域固定元素分组擦除。

## 下一步

固定输入、权重、分组和可重放缺失记录，接入丢失与置零对照；随后扩大训练数据，补齐PSNR/LPIPS，与队员推进量化、真实分包、发送字节统计和期限口径，再验证丢包训练、恢复与交织。

本次仅发布第二周代码与已完成记录，不包含正在运行的500轮实验结果或中间权重。第一周原说明存于[docs/week1_readme.md](docs/week1_readme.md)。
