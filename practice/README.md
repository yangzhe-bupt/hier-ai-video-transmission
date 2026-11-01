# practice目录说明

现有18个练习源码已分组；原文件仅修订头部说明，代码Token与AST均保持一致。

- basics：早期ANN、CNN学习练习。
- data：帧数统计、GOP索引、单视频和目录数据集。
- models：现有CNN编码器、解码器。
- models/swin：窗口注意力、窗口块、移位窗口和Swin块对。
- experiments：单GOP演示、多轮视频训练。
- checks：编解码器、检查点和毕然模块联调。

## 运行方式

请在Anaconda的pytorch环境中使用统一入口，避免直接运行移动后的文件。
原脚本的__file__路径计算按移动前位置兼容，data、outputs、Biran仍在D:/learn/pytorch下。

```powershell
python D:/learn/pytorch/practice/run.py --list
python D:/learn/pytorch/practice/run.py models/swin/swin_block_pair.py
python D:/learn/pytorch/practice/run.py experiments/train_video.py
python D:/learn/pytorch/practice/run.py checks/check_biran_integration.py
```

运行训练脚本仍会执行其原有底部入口，可能重新训练并覆盖原配置指定的检查点。
统一入口只兼容文件布局，不调整模型、训练、数据、参数或原导入语句。
旧__pycache__保留不改动；未创建swin_encoder.py。
