# ======================================================================
# 文件：reconstruct_video.py
# 位置：practice/experiments
# 功能：连续读取视频，按GOP编码解码，保存原始裁剪视频和重建视频。
# 类与函数汇总：
#   - verify_output_video：重新解码输出文件，核对帧数与尺寸。
#   - reconstruct_video：组织读取、预处理、补帧、推理、写出和结果记录。
#   - main：配置输入视频、模型权重和独立输出目录。
# 复用函数：
#   - center_crop_frames、frames_to_gop：来自video_feature_demo。
#   - load_swin_models：来自train_swin_video。
# 流程：视频 → 顺序读GOP → 裁剪与张量转换 → 编码 → 解码 → 视频写出。
# 输入：[1,T,3,128,128]；特征：[1,T,24,32,32]；重建形状与输入一致。
# 输出：original_crop.mp4、reconstructed.mp4、summary.json。
# 运行：python D:/learn/pytorch/practice/run.py experiments/reconstruct_video.py
# 当前状态：无丢包、无音频；中心裁剪演示，尚未接入分包和网络。
# ======================================================================

from datetime import datetime
from pathlib import Path
from time import perf_counter
import json
import math

import cv2
import torch

from video_feature_demo import center_crop_frames, frames_to_gop
from train_swin_video import load_swin_models


# -----------------1.检查实际写出的视频-----------------------
def verify_output_video(video_path, expected_frames, crop_size):
    """
    重新顺序解码输出文件，检查是否可读、帧数和尺寸是否正确。

    参数：
        video_path：已关闭写入器的输出视频路径。
        expected_frames：预期真实帧数，不包含补帧。
        crop_size：预期图像边长。
    返回：dict，包含实际解码帧数、文件标称帧率和文件字节数。
    """
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"输出视频无法打开：{video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (crop_size, crop_size, 3):
                raise RuntimeError(f"输出帧尺寸错误：{frame.shape}")
            count += 1
    finally:
        cap.release()

    if count != expected_frames:
        raise RuntimeError(
            f"输出帧数不一致：预期{expected_frames}，实际{count}；{video_path}"
        )
    return {"frames": count, "fps": fps,
            "file_bytes": Path(video_path).stat().st_size}


# -----------------2.连续视频重建-----------------------
def reconstruct_video(
    video_path, checkpoint_path, output_dir,
    crop_size=128, gop_size=4, device=None, max_frames=None
):
    """
    复用已有预处理模块，连续重建视频并保存运行记录。

    参数：
        video_path：输入视频路径，从开头顺序读取。
        checkpoint_path：Swin编码器＋CNN解码器检查点路径。
        output_dir：尚不存在的新目录，防止覆盖旧结果。
        crop_size：中心裁剪边长；当前窗口4编码器要求为8的倍数且大于8。
        gop_size：每组帧数，默认4。
        device：None时自动选择CUDA/CPU，也可指定"cpu"或"cuda"。
        max_frames：None表示读取至结束；正整数表示只处理前若干帧。
    返回：summary字典，与summary.json内容相同。

    说明：尾部重复最后一帧补齐，推理后仅保存真实帧。
          MSE在视频编码写出前统计，不含MP4编码失真。
          两个MP4均重新编码且无音频，不能作为网络传输码率证据。
    """
    if type(gop_size) is not int or gop_size <= 0:
        raise ValueError("GOP帧数必须为正整数")
    if type(crop_size) is not int or crop_size <= 8 or crop_size % 8:
        raise ValueError("当前编码器要求裁剪边长为8的倍数且大于8")
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError("max_frames必须为正整数或None")

    video_path = Path(video_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not video_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("输入视频或模型检查点不存在")

    device = torch.device(device if device is not None else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder, decoder, checkpoint = load_swin_models(checkpoint_path, device)
    encoder.eval()
    decoder.eval()

    # 根据实际检查点再核对窗口尺寸，兼容后续不同窗口配置。
    window_size = checkpoint["config"]["window_size"]
    if crop_size % (2 * window_size) or crop_size // 2 <= window_size:
        raise ValueError("裁剪尺寸不满足检查点的窗口配置")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    original_path = output_dir / "original_crop.mp4"
    reconstructed_path = output_dir / "reconstructed.mp4"

    cap = cv2.VideoCapture(str(video_path))
    original_writer = reconstructed_writer = None
    total_frames = gop_count = padded_frames = 0
    squared_error = 0.0
    total_elements = 0

    try:
        if not cap.isOpened():
            raise RuntimeError(f"输入视频打开失败：{video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("无法取得有效源视频帧率")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        original_writer = cv2.VideoWriter(
            str(original_path), fourcc, fps, (crop_size, crop_size))
        reconstructed_writer = cv2.VideoWriter(
            str(reconstructed_path), fourcc, fps, (crop_size, crop_size))
        if not original_writer.isOpened() or not reconstructed_writer.isOpened():
            raise RuntimeError("视频写入器创建失败，请检查MP4编码支持及输出目录")

        print("运行设备：", device)
        print("加载最佳轮数：", checkpoint["epoch"])
        print("源视频标称帧率：", fps)
        print("输出目录：", output_dir)
        # 计时不含模型加载、输出文件复查；包含读取、预处理、推理和写出调用。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()

        with torch.no_grad():
            while max_frames is None or total_frames < max_frames:
                wanted = gop_size if max_frames is None else min(
                    gop_size, max_frames - total_frames)
                frames = []
                # 同一个读取对象持续向后读，避免反复从视频开头跳帧。
                for _ in range(wanted):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break

                valid_frames = len(frames)
                frames = center_crop_frames(frames, crop_size)
                padded_frames += gop_size - valid_frames
                while len(frames) < gop_size:
                    frames.append(frames[-1].copy())

                # 复用RGB转换、[0,1]归一化和BTCHW组装。
                gop = frames_to_gop(frames).to(device)
                reconstructed = decoder(encoder(gop))
                if reconstructed.shape != gop.shape:
                    raise RuntimeError("重建形状与输入不一致")
                if not torch.isfinite(reconstructed).all().item():
                    raise RuntimeError("重建结果出现NaN或无穷")

                # 排除补帧，按真实像素数量累计误差。
                difference = reconstructed[:, :valid_frames] - gop[:, :valid_frames]
                squared_error += difference.square().sum().item()
                total_elements += difference.numel()

                # RGB张量 → HWC数组 → uint8图像。
                images = (reconstructed[0, :valid_frames].permute(0, 2, 3, 1)
                          .clamp(0, 1).mul(255).round()
                          .to(torch.uint8).cpu().numpy())
                for t in range(valid_frames):
                    original_writer.write(frames[t])
                    reconstructed_writer.write(
                        cv2.cvtColor(images[t], cv2.COLOR_RGB2BGR))

                total_frames += valid_frames
                gop_count += 1
                if gop_count == 1 or gop_count % 50 == 0:
                    print(f"已处理{gop_count}组，共{total_frames}帧")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        if total_frames == 0:
            raise RuntimeError("没有成功读取任何帧")
    finally:
        # 必须关闭写入器，再重新打开文件核对帧数。
        cap.release()
        if original_writer is not None:
            original_writer.release()
        if reconstructed_writer is not None:
            reconstructed_writer.release()

    verified = {
        "original": verify_output_video(original_path, total_frames, crop_size),
        "reconstructed": verify_output_video(reconstructed_path, total_frames, crop_size)
    }
    for result in verified.values():
        if not math.isclose(result["fps"], fps, rel_tol=1e-3, abs_tol=1e-3):
            raise RuntimeError("写出视频帧率与源视频标称帧率不匹配")

    summary = {
        "video_path": str(video_path), "checkpoint_path": str(checkpoint_path),
        "best_epoch": checkpoint["epoch"], "model_config": checkpoint["config"],
        "device": str(device), "crop_size": crop_size, "gop_size": gop_size,
        "max_frames": max_frames, "decoded_input_frames": total_frames,
        "gops": gop_count, "padding_frames_not_written": padded_frames,
        "source_nominal_fps": fps, "model_mse_before_video_encoding": squared_error / total_elements,
        "processing_seconds_excluding_setup_and_verification": elapsed,
        "processing_fps": total_frames / elapsed,
        "output_videos": verified, "original_path": str(original_path),
        "reconstructed_path": str(reconstructed_path),
        "limitations": ["128x128默认中心裁剪，无音频，无网络",
                        "MP4另有编码损失，输出文件大小不是学习码流发送量",
                        "恒定标称帧率写出，不保留可变帧率时间戳",
                        "输入读取失败即结束；成功解码帧数不保证等于损坏源文件声明帧数"]
    }
    record_path = output_dir / "summary.json"
    record_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("视频重建完成，输出解码帧数与输入成功读取帧数一致：", total_frames)
    print("写出前模型MSE：", f"{squared_error / total_elements:.8f}")
    print("原始裁剪视频：", original_path)
    print("重建视频：", reconstructed_path)
    print("运行记录：", record_path)
    return summary


# -----------------3.运行配置-----------------------
def main():
    """无参数；通过run.py运行，使用第314轮最佳模型重建验证视频。"""
    project_dir = Path(__file__).resolve().parent.parent
    reconstruct_video(
        video_path=project_dir / "data/videos/val/sintel_trailer_720p.mp4",
        checkpoint_path=(project_dir / "outputs/swin_runs"
                         / "20261101_203819_892263/best.pt"),
        output_dir=(project_dir / "outputs/video_reconstruction"
                    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        crop_size=128, gop_size=4,
        # 首次可改为12，只处理前12帧；None表示处理完整视频。
        max_frames=None
    )


if __name__ == "__main__":
    main()
