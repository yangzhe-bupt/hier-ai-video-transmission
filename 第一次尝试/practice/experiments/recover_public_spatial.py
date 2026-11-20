# ======================================================================
# 文件：recover_public_spatial.py
# 功能：用公共分支已收位置的邻域均值逐步填补缺块。
# 输入：反量化后的公共特征与同形状布尔缺失掩码，布局 BTCHW。
# 输出：补全后的特征，已收位置保持原值。
# ======================================================================

import torch
import torch.nn.functional as F


def recover_public_spatial(public, missing_mask):
    """
    用八邻域内已知值的平均数补齐缺失位置。

    参数：
        public：形状为 [B,T,C,H,W] 的浮点公共特征，T 通常为 1。
        missing_mask：同形状、同设备的 bool 张量，True 表示缺失。
    返回：
        与 public 同形状、类型和设备的张量。
        若一个二维平面没有任何已收位置，其缺失位置输出零。
    """
    if public.ndim != 5 or not public.is_floating_point():
        raise ValueError("公共特征必须是 BTCHW 浮点张量")
    if (missing_mask.shape != public.shape or missing_mask.dtype != torch.bool
            or missing_mask.device != public.device):
        raise ValueError("缺失掩码必须为同形状、同设备的 bool 张量")

    height, width = public.shape[-2:]
    # 将批次、时间和通道视为独立平面，避免跨通道混合。
    known = ~missing_mask.reshape(-1, 1, height, width)
    values = public.reshape_as(known).masked_fill(~known, 0)

    # 每轮向缺块内部扩展一圈。卷积窗口中心为未知位置，
    # 因此在该位置统计到的已知值实际只来自周围八格。
    for _ in range(max(height, width) - 1):
        if known.all().item():
            break
        weight = F.avg_pool2d(known.to(values.dtype), 3, stride=1, padding=1,
                              divisor_override=1)
        fillable = ~known & (weight > 0)
        if not fillable.any().item():
            break
        total = F.avg_pool2d(values, 3, stride=1, padding=1, divisor_override=1)
        values = torch.where(fillable, total / weight.clamp_min(1), values)
        known = known | fillable
    return values.reshape_as(public)


if __name__ == "__main__":
    # 缺失位置故意使用非零占位值。中心应等于已知邻居 2 和 4 的均值。
    values = torch.tensor([2., 99., 4.]).reshape(1, 1, 1, 1, 3)
    mask = values == 99
    expected = torch.tensor([2., 3., 4.]).reshape_as(values)
    torch.testing.assert_close(recover_public_spatial(values, mask), expected)
    torch.testing.assert_close(
        recover_public_spatial(values, torch.ones_like(mask)), torch.zeros_like(values),
    )
    print("通过：邻域均值、已收元素保留和全缺回退。")
