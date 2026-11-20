# 开放视频调试素材集

用于Dataset/DataLoader及轻量编解码流程调试，不是正式论文数据集。每个集合仅一部独立影片，规模与内容覆盖不足以支撑正式性能结论；视频为有损发行版本，不是原始无损源。

- train：Big Buck Bunny官方720p预告片，CC BY 3.0。署名：(c) copyright 2008, Blender Foundation / www.bigbuckbunny.org。许可：https://peach.blender.org/about/
- val：Sintel官方720p预告片，CC BY 3.0。署名：Blender Foundation | durian.blender.org。许可：https://durian.blender.org/sharing/
- test：Tears of Steel官方720p完整发行版，CC BY 3.0。署名：(CC) Blender Foundation | mango.blender.org。许可：https://mango.blender.org/about/

保留影片原始文件和片尾署名。训练仅使用train；val用于选模型；test在配置确定后评测，不用于训练或反复调参。来源、大小、SHA256见sources.json；帧数及解码检查见validation.json。

文件后缀包含.mov和.mp4，扫描代码应同时支持。实际帧率分别测量，不强制改成24fps；本素材仅用于当前流程，不能据此宣称720P@24fps约1Mbps目标已达成。
