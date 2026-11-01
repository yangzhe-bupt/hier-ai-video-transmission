# ======================================================================
# 文件：video_dataset.py
# 位置：practice/data
# 功能：将单个视频组织为GOP样本，并检查Dataset、DataLoader及编解码器衔接。
# 类与函数汇总：
#   - VideoGOPDataset：将一个视频组织为多个GOP样本
#     方法：__init__、__len__、__getitem__
#   - check_video_dataset：检查样本数量，形状以及样本编号对应关系
#   - check_video_dataloader：检查多个GOP能否组成一个batch
#   - check_batch_model：检查真实批量GOP与编解码器的形状及数值衔接。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py data/video_dataset.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

import torch
from torch.utils.data import Dataset,DataLoader
from semantic_encoder import GOPEncoder
from semantic_decoder import GOPDecoder
from gop_index import build_gop_starts,count_video_frames
from video_feature_demo import load_video_as_gop

#----------------1.构建视频GOP数据集------------------
class VideoGOPDataset(Dataset):
    """
    将一个视频组织为多个GOP样本

    参数：
        video_path:视频路径
        gop_size:每个样本帧数
        stride:相邻样本起点间隔
        crop_size:裁剪边长

    单个样本：
        gop:[T,C,H,W]
        start_frame:该GOP的起始帧编号
    """
    def __init__(self,video_path,gop_size=4,stride=4,crop_size=128):
        super().__init__()

        self.video_path = video_path
        self.gop_size = gop_size
        self.crop_size = crop_size

        total_frames = count_video_frames(video_path)

        self.starts = build_gop_starts(
            total_frames,
            gop_size=gop_size,
            stride=stride
        )

        if not self.starts:
            raise ValueError("视频无法组成一个完整GOP")

    def __len__(self):
        """返回样本数量"""
        return len(self.starts)

    def __getitem__(self, index):
        """
        按照编号读取一个GOP
        
        参数：
            index:样本编号

        返回：
            dict:GOP张量以及对于起始帧编号
        """
        start_frame = self.starts[index]

        gop = load_video_as_gop(
            self.video_path,
            crop_size=self.crop_size,
            num_frames=self.gop_size,
            start_frame=start_frame
        )

        # 原函数返回[1,T,3,H,W],Dataset每次返回一个样本，因此去掉最前面的1
        gop = gop.squeeze(0)

        return {
            "gop":gop,
            "start_frame":start_frame
        }

#----------------2.检查数据集--------------------------
def check_video_dataset():
    """
    检查样本数量，形状以及样本编号对应关系
    """
    #创建数据集
    dataset = VideoGOPDataset(video_path=r"D:\learn\pytorch\data\毕洋洋.mp4")
    #取样本编号1，即第二组，帧4~7
    sample = dataset[1]

    assert sample["start_frame"] == 4
    assert sample["gop"].shape == (4, 3, 128, 128)

    print("数据集样本数：", len(dataset))
    print("样本1起始帧：", sample["start_frame"])
    print("单样本形状：", sample["gop"].shape)
    print("Dataset检查通过")

#---------------3.检查批量加载-----------------------
def check_video_dataloader():
    """
    检查多个GOP能否组成一个batch

    参数：
        无
    
    返回：
        无
        检查失败时抛出AssertionError
    """
    dataset = VideoGOPDataset(video_path=r"D:\learn\pytorch\data\毕洋洋.mp4",gop_size=4,stride=4,crop_size=128)

    # DataLoader：从Dataset取样，并组合成批次
    loader = DataLoader(
        dataset,
        batch_size=2,      # 每批包含2个GOP
        shuffle=False,     # 暂不打乱，方便检查编号
        num_workers=0,     # 在当前进程读取，方便Windows下调试
        drop_last=False    # 保留最后不足2个样本的批次
    )

    # iter(loader)：创建批次迭代器
    # next(...)：取出第一个批次
    batch = next(iter(loader))

    # Dataset返回[T,C,H,W]
    # DataLoader在最前面增加batch维度
    gops = batch["gop"]
    start_frames = batch["start_frame"]

    assert gops.shape == (2, 4, 3, 128, 128)

    # 未打乱时，前两个GOP分别从帧0和帧4开始
    assert start_frames.tolist() == [0, 4]

    print("批次总数：", len(loader))
    print("批量GOP形状：", gops.shape)
    print("对应起始帧：", start_frames.tolist())
    print("DataLoader检查通过")

#--------------4.检查批量数据与编解码器的衔接-----------
def check_batch_model():
    """
    将DataLoader生成的真实皮此次送入编解码器

    参数：
        无
    
    返回：
        无
        检查失败时抛出AssertionError
    """
    dataset = VideoGOPDataset(
        video_path=r"D:\learn\pytorch\data\毕洋洋.mp4",
        gop_size=4,
        stride=4,
        crop_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    batch = next(iter(loader))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 将整个批次移到模型所在设备
    gops = batch["gop"].to(device)

    #编解码器
    encoder = GOPEncoder(embed_dim=24).to(device)
    decoder = GOPDecoder(feature_channels=24).to(device)

    encoder.eval()
    decoder.eval()

    # 本次只检查前向流程，不记录梯度
    with torch.no_grad():
        features = encoder(gops)
        reconstructed = decoder(features)

    assert features.shape == (2, 4, 24, 32, 32)
    assert reconstructed.shape == gops.shape
    assert torch.isfinite(features).all().item()
    assert torch.isfinite(reconstructed).all().item()

    print("运行设备：", device)
    print("批量输入：", gops.shape)
    print("批量特征：", features.shape)
    print("批量重建：", reconstructed.shape)
    print("真实批量数据与模型衔接检查通过")


#主程序
def main():
    check_batch_model()

#测试
if __name__=='__main__':
    main()