"""Step05：读取真实视频的连续四帧，并进行一致裁剪。"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def read_frames(video_path, start_frame=100, num_frames=4):
    """
    从视频中顺序读取指定帧。

    参数：
        start_frame：起始帧编号，从0开始。
        num_frames：连续读取多少帧。

    返回：
        BGR 图像列表，每张图像的布局为 [H,W,C]。
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在：{video_path}")

    if start_frame < 0 or num_frames <= 0:
        raise ValueError("起始帧必须非负，读取帧数必须为正")
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开视频，请检查路径或视频编码")

    # 顺序跳过前面的帧，便于准确理解帧编号。
    # 当前只读一个 GOP，顺序读取的开销可以接受。
        for _ in range(start_frame):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError("视频长度不足，无法到达指定起始帧")

        frames = []

        for index in range(num_frames):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"读取第 {start_frame + index} 帧失败"
                )

            frames.append(frame)

    finally:
        # 成功或发生异常，都释放视频文件资源。
        cap.release()

    return frames

def frames_to_gop(frames, crop_size=256, seed=42):
    """
    对所有帧使用同一个随机裁剪位置，并转换为 RGB 视频张量。

    返回：
        gop：[1,T,3,crop_size,crop_size]，float32。
        crop_position：(top,left)，方便核对裁剪位置。
    """
    if not frames:
        raise ValueError("输入帧列表不能为空")

    if crop_size <= 0:
        raise ValueError("裁剪尺寸必须为正")

    height, width, channels = frames[0].shape

     # 同一 GOP 中各帧的原始尺寸应该相同。
    for frame in frames:
        if frame.shape != (height, width, channels):
            raise ValueError("各帧的原始尺寸不一致")

    if height < crop_size or width < crop_size:
        raise ValueError("视频高或宽小于裁剪尺寸")

    # 只生成一次裁剪坐标，随后用于全部帧。
    # 固定种子，使这次检查能够重复。
    rng = np.random.default_rng(seed)

    # integers 的上界不包含在采样范围内，因此加1。
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))

    tensor_frames = []

    for frame in frames:
        cropped = frame[
            top:top + crop_size,
            left:left + crop_size,
            :,
        ]

        # OpenCV 默认读取 BGR；显示和模型输入统一使用 RGB。
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # uint8 的 [0,255] 转为 float32 的 [0,1]。
        tensor = torch.from_numpy(rgb).float() / 255.0

        # [H,W,C] → [C,H,W]。
        tensor = tensor.permute(2, 0, 1)

        tensor_frames.append(tensor)

    # stack：[T,C,H,W]。
    # unsqueeze：[1,T,C,H,W]，新增批次维。
    gop = torch.stack(tensor_frames, dim=0).unsqueeze(0)

    return gop, (top, left)

def main():
    video_path = Path(
        r"D:\学习与项目\学习代码\pytorch\data\videos"
        r"\train\big_buck_bunny_trailer_720p.mov"
    )

    start_frame = 100
    crop_size = 256

    frames = read_frames(
        video_path,
        start_frame=start_frame,
        num_frames=4,
    )

    gop, crop_position = frames_to_gop(
        frames,
        crop_size=crop_size,
        seed=42,
    )

    print("原始单帧形状 HWC：", frames[0].shape)
    print("共同裁剪位置 top、left：", crop_position)
    print("GOP 形状 BTCHW：", tuple(gop.shape))
    print("数据类型：", gop.dtype)
    print("像素范围：", gop.min().item(), gop.max().item())

    assert gop.shape == (1, 4, 3, crop_size, crop_size)
    assert gop.dtype == torch.float32
    assert torch.isfinite(gop).all()
    assert gop.min() >= 0 and gop.max() <= 1

    # 四列依次显示连续四帧。
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))

    for index in range(4):
        # Matplotlib 需要 HWC 布局。
        image = gop[0, index].permute(1, 2, 0).numpy()

        axes[index].imshow(image)
        axes[index].set_title(f"Frame {start_frame + index}")
        axes[index].axis("off")

    figure.tight_layout()

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "step05_real_gop.png"
    figure.savefig(image_path, dpi=150)

    print("四帧对比图已保存到：", image_path)

    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()