# 分层传输实验资料（2026-11-17）

各目录保存可复核的实验统计、时间线和阶段报告。JSON 中原始本地绝对路径已去除，保留文件名、参数和已有校验值。源视频、权重、演示 MP4、逐数据报日志和中间张量保留在实验机。原始视频来源参见 [datasets](../../docs/datasets/README.md)。

| 主题 | 入口 |
| --- | --- |
| 公共保护和突发丢包 | [公共保护](public_protection_20261116/REPORT.md)、[突发丢包](public_protection_burst_20261116/REPORT.md) |
| 低码率公共档及残差增强 | [公共档](public_base_budget_20261116/REPORT.md)、[残差增强](residual_enhancement_20261116/REPORT.md) |
| MTU 和自适应播放 | [MTU 预算](layered_mtu1500_20261117/REPORT.md)、[自适应播放](adaptive_mtu1500_video_20261117/REPORT.md) |
| 本机 UDP 和实时接收 | [回环](udp_loopback_20261117/REPORT.md)、[在线接收对照](live_udp_receiver_compare_20261117/REPORT.md) |
| 全画面传输与画质 | [传输汇总](full_frame_udp_20261117/summary.json)、[指标汇总](full_frame_quality_20261117/summary.csv)、[逐帧结果](full_frame_quality_20261117/per_frame.csv) |

各阶段的分辨率、丢包粒度和时钟模型不同。全画面 720P 尺寸输出来自 128×72 缩放后的编码，LPIPS 和原生 720P 验收待测。
