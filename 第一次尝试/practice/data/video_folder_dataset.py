# ======================================================================
# 文件：video_folder_dataset.py
# 位置：practice/data
# 功能：扫描视频目录，建立跨视频GOP索引，提供批量加载检查与中间样本查看。
# 类与函数汇总：
#   - VideoFolderDataset：扫描一个目录，将多个视频的GOP组织为一个数据集
#     方法：__init__、__len__、__getitem__
#   - check_folder_loader：检查视频目录能否通过DataLoader生成一个批次
#   - check_middle_gop：查看数据集中间一个GOP的内容。
#   - main：运行底部入口当前启用的检查或演示。
# 运行：python D:/learn/pytorch/practice/run.py data/video_folder_dataset.py
# 路径兼容：通过run.py运行；原导入、函数体与路径表达式保持不变。
# ======================================================================

from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from video_dataset import VideoGOPDataset

#-------------1.多视频GOP数据集-------------------
class VideoFolderDataset(Dataset):
    """
    扫描一个目录，将多个视频的GOP组织为一个数据集

    参数：
        video_dir:视频所在目录
        gop_size:每个GOP帧数
        stride:相邻起点间隔帧数
        crop_size:裁剪边长

    单个样本：
        gop:[T,C,H,W]
        start_frame:原视频中的起始帧编号
        video_name:来源视频文件名
    """
    def __init__(
            self,
            video_dir,
            gop_size=4,
            stride=4,
            crop_size=128
        ):
        super().__init__()

        folder = Path(video_dir)

        if not folder.is_dir():
            raise ValueError(f"目录不存在：{folder}")

        # iterdir()：遍历目录中的文件
        # suffix.lower()：读取后缀，并统一为小写
        # sorted()：固定文件顺序，方便重复实验
        video_paths = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".mp4", ".mov", ".mkv"}
        )

        if not video_paths:
            raise ValueError("目录中没有支持的视频文件")

        self.datasets = []
        self.video_names = []

        self.sample_index = []

        # 遍历扫描得到的视频路径列表
        for video_id, path in enumerate(video_paths):
            dataset = VideoGOPDataset(
                video_path=str(path),
                gop_size=gop_size,
                stride=stride,
                crop_size=crop_size
            )

            self.datasets.append(dataset)
            self.video_names.append(path.name)

            for local_index in range(len(dataset)):
                self.sample_index.append(
                    (video_id, local_index)
                )

    def __len__(self):
        """返回所有视频的GOP总数。"""
        return len(self.sample_index)

    def __getitem__(self, index):
        """
        根据全局样本编号，读取对应视频中的一个GOP

        参数：
            index:数据集中的样本编号

        返回:
            dict:
                gop:视频张量[T,C,H,W]
                start_frame:起始帧编号
                video_name:来源视频文件名
        """
        # 将全局编号转换为：视频编号、视频内部GOP编号
        video_id, local_index = self.sample_index[index]

        # 找到对应的单视频数据集，再读取其中的GOP
        sample = self.datasets[video_id][local_index]

        return {
            "gop": sample["gop"],
            "start_frame": sample["start_frame"],
            "video_name": self.video_names[video_id]
        }

#-------------------2.检查文件夹数据加载-----------------------
def check_folder_loader(video_dir,split_name):
    """
    检查视频目录能否通过DataLoader生成一个批次

    参数:
        video_path:视频文件夹路径
        split_name:打印用的名称
    """
    dataset = VideoFolderDataset(
        video_dir=video_dir,
        gop_size=4,
        stride=4,
        crop_size=128
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 取出第一个批次
    batch = next(iter(loader))
    gops = batch["gop"]

    # 样本不足2个时，实际批量大小会小于2
    batch_size = min(2, len(dataset))

    assert gops.shape == (batch_size, 4, 3, 128, 128)
    assert gops.dtype == torch.float32
    assert torch.isfinite(gops).all().item()
    assert gops.min().item() >= 0
    assert gops.max().item() <= 1

    print(f"\n===== {split_name} =====")
    print("视频数量：", len(dataset.video_names))
    print("视频名称：", dataset.video_names)
    print("GOP总数：", len(dataset))
    print("批次总数：", len(loader))
    print("批量形状：", gops.shape)
    print("来源视频：", batch["video_name"])
    print("起始帧：", batch["start_frame"].tolist())
    print(f"{split_name}数据加载检查通过")

def check_middle_gop(video_dir):
    """
    查看数据集中间一个GOP的内容。

    参数：
        video_dir：视频文件夹路径

    返回：
        无；打印样本信息并显示第一帧
    """
    import matplotlib.pyplot as plt

    dataset = VideoFolderDataset(video_dir=video_dir)

    # 取中间位置，避免只检查片头
    index = len(dataset) // 2
    sample = dataset[index]
    gop = sample["gop"]  # [T,3,H,W]

    print("样本编号：", index)
    print("来源视频：", sample["video_name"])
    print("起始帧：", sample["start_frame"])
    print("数值范围：", gop.min().item(), gop.max().item())

    # 显示该GOP的第一帧：[C,H,W] → [H,W,C]
    plt.imshow(gop[0].permute(1, 2, 0).numpy())
    plt.title(f"Start frame: {sample['start_frame']}")
    plt.axis("off")
    plt.show()

#主程序
def main():
    """分别检查训练集和验证集的数据加载。"""
    # 当前文件上一级目录为 D:\learn\pytorch
    project_dir = Path(__file__).resolve().parent.parent
    videos_dir = project_dir / "data" / "videos"

    check_folder_loader(
        video_dir=videos_dir / "train",
        split_name="train"
    )

    check_folder_loader(
        video_dir=videos_dir / "val",
        split_name="val"
    )
    check_middle_gop(videos_dir / "val")

#测试
if __name__ == "__main__":
    main()