# REDS Sharp下载与校验

仓库不上传数据集。使用作者原始训练/验证划分，不将相邻帧随机拆到两个集合。

- [作者官网](https://seungjunnah.github.io/Datasets/reds.html)
- [作者Hugging Face文件列表](https://huggingface.co/datasets/snah/REDS/tree/main)
- 仅需`train_sharp.zip`与`val_sharp.zip`，不下载blur/bicubic/120fps包。
- 数据页面许可证CC BY 4.0，使用时保留来源并遵循许可。

固定revision：`62dc25d16e6f43d2214f1b365023abda86f7a0ae`。

| 文件 | 下载地址 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| train_sharp.zip | [下载](https://huggingface.co/datasets/snah/REDS/resolve/62dc25d16e6f43d2214f1b365023abda86f7a0ae/train_sharp.zip?download=true) | 34261573976 | `620294c1c3f23ed26c5ea228633770469c0b28e57d31eb41dc77deb401c6681b` |
| val_sharp.zip | [下载](https://huggingface.co/datasets/snah/REDS/resolve/62dc25d16e6f43d2214f1b365023abda86f7a0ae/val_sharp.zip?download=true) | 4273865704 | `53f6ce4791ed3cfca747ff8982ba29b6e8b7cc1927eed742458f45ed93a1a279` |

```powershell
Get-FileHash 'D:\Datasets\REDS\train_sharp.zip' -Algorithm SHA256
Get-FileHash 'D:\Datasets\REDS\val_sharp.zip' -Algorithm SHA256
```

将两包放在同一个目录，通过`--data-root`传入；不需要解压。训练240段、验证30段，每段100帧720×1280。每4帧组成非重叠GOP，训练/验证采用256裁剪。
