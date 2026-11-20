# ======================================================================
# 文件：recover_individual_residual.py
# 功能：利用 factor=1 分解的时间零和约束补齐个体残差。
# 函数汇总：
#   - recover_individual_residual：按时间维估计缺失元素。
#   - check_recovery：用手算样例验证四种缺失情况。
# 输入：反量化后的 BTCHW 个体残差及同形状布尔掩码。
# 输出：已收到的元素保持原值，缺失位置填入估计值。
# 运行：python practice/run.py experiments/recover_individual_residual.py
# ======================================================================

import torch


def recover_individual_residual(individual, missing_mask):
    """
    根据同一 GOP 内其他帧的已收残差，估计缺失残差。

    参数：
        individual：反量化后的浮点残差，形状为 [B,T,C,H,W]。
        missing_mask：同形状、同设备的布尔张量，True 表示缺失。

    返回：
        与输入同形状、类型和设备的恢复残差。
    """
    if individual.ndim != 5 or not individual.is_floating_point():
        raise ValueError("个体残差必须是 [B,T,C,H,W] 浮点张量")
    if missing_mask.shape != individual.shape or missing_mask.dtype != torch.bool:
        raise ValueError("缺失掩码必须是与残差同形状的 bool 张量")
    if missing_mask.device != individual.device:
        raise ValueError("残差和掩码必须位于同一设备")

    # 公共特征取自时间均值，因此量化前同一位置的时间残差和约为零。
    # 估计时先屏蔽缺失占位值，避免把占位值误当作观测。
    observed_sum = individual.masked_fill(missing_mask, 0).sum(dim=1, keepdim=True)
    missing_count = missing_mask.sum(dim=1, keepdim=True)

    # 缺一个时直接补差额，缺多个时均分差额。整列全缺时和为零。
    # 无缺失时将分母下限设为 1，实际输出仍由原值提供。
    estimated = -observed_sum / missing_count.clamp_min(1)
    return torch.where(missing_mask, estimated, individual)


def check_recovery():
    """检查缺一帧、缺两帧、全缺及无缺四种情况。"""
    # 四列分别表示四个空间位置。99 是故意设置的缺失占位值。
    individual = torch.tensor([
        [99.0,  2.0, 99.0,  1.0],
        [ 2.0, 99.0, 99.0, -1.0],
        [-1.0,  4.0, 99.0,  2.0],
        [-3.0, 99.0, 99.0, -2.0],
    ]).reshape(1, 4, 1, 1, 4)
    missing_mask = individual == 99
    expected = torch.tensor([
        [ 2.0,  2.0, 0.0,  1.0],
        [ 2.0, -3.0, 0.0, -1.0],
        [-1.0,  4.0, 0.0,  2.0],
        [-3.0, -3.0, 0.0, -2.0],
    ]).reshape_as(individual)
    recovered = recover_individual_residual(individual, missing_mask)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)
    print("通过：单帧缺失、多帧缺失、全缺失与无缺失。")


if __name__ == "__main__":
    check_recovery()
