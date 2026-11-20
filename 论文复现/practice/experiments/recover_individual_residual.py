# ======================================================================
# 文件：recover_individual_residual.py
# 位置：practice/experiments
# 功能：利用 factor=1 分解的残差时间和约束，估计缺失的个体残差。
#
# 函数汇总：
#   - recover_individual_residual：按时间维补齐缺失残差。
#   - check_recovery：检查单帧缺失、多帧缺失、全缺失和无缺失。
#
# 输入：
#   individual：接收端反量化后的个体残差 [B,T,C,H,W]。
#   missing_mask：同形状 bool 掩码，True 表示对应元素缺失。
#
# 输出：
#   与 individual 同形状的恢复残差，已收到的元素保持原值。
#
# 运行：
#   python practice/run.py experiments/recover_individual_residual.py
# ======================================================================

import torch


def recover_individual_residual(individual, missing_mask):
    """
    用已收到的个体残差，估计同一位置在其他帧中的缺失残差。

    参数：
        individual：反量化后的接收残差，形状为 [B,T,C,H,W]。
        missing_mask：同形状的布尔掩码，True 表示缺失。

    返回：
        recovered：恢复后的个体残差，形状、设备和类型与输入一致。
    """
    if individual.ndim != 5 or not individual.is_floating_point():
        raise ValueError("个体残差必须是 [B,T,C,H,W] 浮点张量")

    if missing_mask.shape != individual.shape or missing_mask.dtype != torch.bool:
        raise ValueError("缺失掩码必须是与残差同形状的 bool 张量")

    if missing_mask.device != individual.device:
        raise ValueError("残差和缺失掩码必须在同一设备上")

    # 计算已收残差的和时，将缺失位置排除。
    # 即使输入中缺失位置存着非零占位值，也不会参与估计。
    observed = individual.masked_fill(missing_mask, 0)
    observed_sum = observed.sum(dim=1, keepdim=True)

    # 只沿时间维统计缺失数，结果为 [B,1,C,H,W]。
    # 每个位置都用自己的缺失数，不使用整个 GOP 的丢失包数。
    missing_count = missing_mask.sum(dim=1, keepdim=True)

    # 估计缺失残差的总和，再均分到缺失的帧。
    # clamp_min(1) 避免无缺失位置除以零。
    estimated = -observed_sum / missing_count.clamp_min(1)

    # 如果同一位置所有帧都缺失，observed_sum 为零，估计值也为零。
    # 广播时间维后，仅替换掩码为 True 的位置。
    recovered = torch.where(missing_mask, estimated, individual)
    return recovered


def check_recovery():
    """用四个空间位置分别检查四种缺失情况。"""
    # 行对应帧，列对应空间位置。
    # 从左到右分别为：缺一帧、缺两帧、全缺失、无缺失。
    # 99 是故意填入的缺失占位值，用来检查它是否被正确排除。
    individual = torch.tensor([
        [99.0,  2.0, 99.0,  1.0],
        [ 2.0, 99.0, 99.0, -1.0],
        [-1.0,  4.0, 99.0,  2.0],
        [-3.0, 99.0, 99.0, -2.0],
    ]).reshape(1, 4, 1, 1, 4)

    missing_mask = individual == 99.0
    recovered = recover_individual_residual(individual, missing_mask)

    expected = torch.tensor([
        [ 2.0,  2.0, 0.0,  1.0],
        [ 2.0, -3.0, 0.0, -1.0],
        [-1.0,  4.0, 0.0,  2.0],
        [-3.0, -3.0, 0.0, -2.0],
    ]).reshape_as(individual)

    torch.testing.assert_close(recovered, expected)
    assert torch.equal(recovered[~missing_mask], individual[~missing_mask])
    print("通过：单帧缺失、多帧缺失、全缺失和已收元素保留检查。")


if __name__ == "__main__":
    check_recovery()