# PING-III 层次化AI视频传输

面向动态网络的语义视频AI层次化编解码与丢包鲁棒自适应传输研究。

当前交付是**轻量、逐帧CNN无丢失基线**和独立的**特征域分组擦除模块**，用于学习、接口联调和机制验证。完整分层、语义压缩链、真实分包、恢复网络与网络自适应尚未在本仓库闭环；不代表GRACE或MSTVSC完整复现。

## 目标与分工

阶段目标为720P、24fps源视频、约1Mbps实际总发送码率、10%丢包和LPIPS-AlexNet < 0.6。以上均待验证；源帧率与实际处理速度分别报告，码率需声明包头、元信息、保护及反馈的统计范围。

|成员|当前分工|
|---|---|
|杨哲|轻量语义编解码主干；缺失掩码、丢包训练、恢复和鲁棒性评测|
|毕然|公共/个体分解、下采样、量化、压缩逆变换、分包和交织；RDO|
|蒋立果|信道、轨迹、预算、期限、反馈与网络自适应；分层接口由团队进一步确认|

分层表示的生成与层次发送选择需另行约定。公共/个体分解不自动等于基础/增强层，当前模型没有可选传输层或帧间预测。

## 目录

```text
hier_video/
  models/semantic_encoder.py    # PatchEmbedding、降采样、GOPEncoder
  models/semantic_decoder.py    # 上采样、GOPDecoder
  data/video.py                # 指定起始帧、中心裁剪、RGB与BTCHW组装
  transmission/packet_loss.py  # 特征分组独立擦除
  checkpoint.py               # 保存与加载模型，不依赖检查程序
  experiments/
    train_single_gop.py        # 固定GOP过拟合检查，显式命令启动
    evaluate.py                # 只加载权重，保存CSV和重建图
checks/
  check_packet_loss.py         # CPU/CUDA丢失、掩码与输入不变检查
  check_model_components.py    # 各层形状、逐帧对应与梯度检查
  check_autoencoder.py         # 联合梯度、更新、尺寸及检查点往返
packet_loss/                  # 旧包导入兼容层，不维护重复实现
context                       # 周成果、结果边界及待对接事项
requirements.txt
```

本地`data/`、`outputs/`、视频、权重及特征样例默认不提交。原D盘学习文件不受此次重构影响，原`video_feature_demo.py`的职责已拆为数据、训练、评测和检查点模块。

## 环境

已验证环境：Conda、PyTorch 2.6.0+cu124、RTX 4060 Laptop GPU（约8GB）。CPU也可运行基础检查。GPU版PyTorch请根据本机环境安装，保留已有可用环境即可。

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

以下命令均在仓库根目录、已激活的Python环境执行。

## 基础检查

```bash
python -m checks.check_packet_loss
python -m checks.check_model_components
python -m checks.check_autoencoder
```

丢包与联合编解码检查会在CPU运行，CUDA可用时再检查GPU。组件检查不代表系统性能达标。旧包入口可使用`python -m packet_loss.check_packet_loss`；不要进入子目录直接运行旧文件。

## 训练和评测

准备一段可用于实验的视频，裁剪尺寸需为正且能被4整除。训练只使用指定GOP，是流程检查，不是多视频训练。

```bash
python -m hier_video.experiments.train_single_gop --video data/sample.mp4 --checkpoint outputs/single_gop_v1.pt --steps 5000 --start-frame 0 --crop-size 128
```

默认种子24，学习率0.001，4帧GOP、24特征通道、MSE损失。检查点已存在时拒绝覆盖，请选择新文件名。可用`--device cpu`或`--device cuda`选择设备。

```bash
python -m hier_video.experiments.evaluate --video data/sample.mp4 --checkpoint outputs/single_gop_v1.pt --start-frames 0 100 --output-dir outputs/evaluation
```

评测不训练、不覆盖权重，输出`metrics.csv`及每个片段的原图/重建对比PNG，不需要弹窗。视频需足够长；帧号从0开始。使用自己的旧权重时，替换`--checkpoint`路径。默认不分发原训练视频和权重，不能保证换视频复现相同数值。

新检查点记录训练配置、种子、模型权重及损失；未保存优化器状态，因此不是完整断点续训。旧权重参数名保持兼容。

## 编解码接口

```python
from hier_video.models.semantic_encoder import GOPEncoder
from hier_video.models.semantic_decoder import GOPDecoder

encoder = GOPEncoder(embed_dim=24)
decoder = GOPDecoder(feature_channels=24)
features = encoder(gop)
reconstructed = decoder(features)
```

|对象|格式|说明|
|---|---|---|
|输入视频|`[B,T,3,H,W]` float32|RGB、[0,1]；H/W能被4整除|
|编码特征|`[B,T,24,H/4,W/4]` float32|可为负值，尚未量化|
|重建视频|`[B,T,3,H,W]` float32|Sigmoid输出[0,1]|

当前各帧共享参数但独立计算，无帧间或跨GOP依赖，没有`layer_id`、真实包大小或码率档位。模型与输入需处于同一设备；训练直接调用`encoder(gop)`，不要调用使用`no_grad()`的`encode_gop_for_inference()`。

## 丢包模块接口

```python
from hier_video.transmission.packet_loss import simulate_packet_loss
received, mask, dropped = simulate_packet_loss(features, p=0.1, packet_size=16)
```

`packet_size`单位为**元素**，不是字节；整个张量统一展平，因此当前可能跨样本/GOP分组。元素总数必须整除包大小。`mask`与特征同形状、bool、True=缺失；`dropped`为一维包级记录。每组独立采样，实际丢失比例不保证等于p。

尚无量化、包头、轨迹持久化、突发丢失或迟到处理，也未与编解码训练入口联接。未来由毕然提供映射、蒋立果提供包事件，杨哲据此构造掩码，避免重复随机丢包。

## 已有结果与边界

杨哲报告的同一视频128×128中心裁剪实验：仅在帧0～3训练5000步，最后训练日志MSE约0.000165；加载后帧0～3为0.000167、帧100～103为0.000519，加载前后输出最大差值0。

这些是原学习流程的实验记录，不是本次重构重新训练所得。当前使用单视频、无量化、无丢包，不证明独立视频泛化、LPIPS、720P、1Mbps或分层能力。详细文字记录见[context](context)。

## 后续协作

先确认特征布局、版本、分组边界与恢复位置；每周提交代码、配置、固定样例、运行命令和检查结果。先回归已验证版本，再单项替换新模块。团队接口融合仍待确认，个人代码完成不等于整周联合验收完成。

方法参考：GRACE（鲁棒训练与独立分包）及MSTVSC（语义表示、交织与恢复）。论文借鉴、部分机制实现与完整复现分别报告。
