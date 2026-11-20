# ======================================================================
# 文件：summarize_residual_recovery.py
# 功能：核对恢复实验与置零基线，保存跨种子的配对统计。
# 函数汇总：
#   - read_rows：读取一个实验条件的逐 GOP 指标。
#   - summarize：核对包事件与置零指标，计算每档丢包率的恢复效果。
#   - main：读取两个结果目录，生成 recovery_summary.csv。
# ======================================================================

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean


def read_rows(path):
    """读取 UTF-8 CSV，按记录顺序返回逐 GOP 字典列表。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(baseline_root, recovery_root):
    """逐条件核对包事件和旧指标，再汇总失败率、配对 MSE 与改善次数。"""
    cases = sorted(p.name for p in baseline_root.glob("loss_*") if p.is_dir())
    recovery_cases = sorted(p.name for p in recovery_root.glob("loss_*") if p.is_dir())
    if not cases or cases != recovery_cases:
        raise ValueError("恢复实验与置零基线的条件目录不一致")
    grouped = defaultdict(list)
    for case in cases:
        baseline = baseline_root / case
        recovery = recovery_root / case
        if (baseline / "events.json").read_bytes() != (recovery / "events.json").read_bytes():
            raise ValueError(f"{case} 的包事件与基线不一致")
        original_rows = read_rows(baseline / "gop_metrics.csv")
        recovery_rows = read_rows(recovery / "gop_metrics.csv")
        if not original_rows or len(original_rows) != len(recovery_rows):
            raise ValueError(f"{case} 的 GOP 数不一致")
        for original, result in zip(original_rows, recovery_rows):
            # 允许 GPU 重跑中的微小浮点差异，状态和样本标识须完全一致。
            for key, value in original.items():
                if value == result[key]:
                    continue
                if key in ("no_loss_int8_mse", "packet_loss_mse", "delta_mse"):
                    if value and result[key] and math.isclose(
                        float(value), float(result[key]), rel_tol=1e-5, abs_tol=1e-8
                    ):
                        continue
                raise ValueError(f"{case} 的 GOP {original['gop_id']} 字段 {key} 不一致")
            grouped[float(result["target_loss_fraction"])].append(result)

    summary = []
    for rate, rows in sorted(grouped.items()):
        failed = sum(row["decoder_ready"] == "False" for row in rows)
        # 只有可解码且确实缺失个体残差的样本进入恢复质量统计。
        paired = [row for row in rows if int(row["missing_individual_elements"] or 0) > 0]
        delta = [float(row["mse_reduction_vs_zero_fill"]) for row in paired]
        summary.append({
            "target_loss_fraction": rate,
            "gop_trials": len(rows),
            "failed_gop_count": failed,
            "failed_gop_fraction": failed / len(rows),
            "paired_recovery_gops": len(paired),
            "improved_gops": sum(x > 1e-8 for x in delta),
            "worsened_gops": sum(x < -1e-8 for x in delta),
            "unchanged_gops": sum(abs(x) <= 1e-8 for x in delta),
            "paired_no_loss_mse": mean(float(r["no_loss_int8_mse"]) for r in paired) if paired else None,
            "paired_zero_fill_mse": mean(float(r["packet_loss_mse"]) for r in paired) if paired else None,
            "paired_recovered_mse": mean(float(r["recovered_mse"]) for r in paired) if paired else None,
            "mean_mse_reduction": mean(delta) if delta else None,
        })
    return summary


def main():
    """从命令行读取基线与恢复目录，生成可复查的配对汇总。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("recovery", type=Path)
    args = parser.parse_args()
    rows = summarize(args.baseline, args.recovery)
    output = args.recovery / "recovery_summary.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
