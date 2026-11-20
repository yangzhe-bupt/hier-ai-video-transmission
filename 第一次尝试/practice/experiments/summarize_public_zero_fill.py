# ======================================================================
# 文件：summarize_public_zero_fill.py
# 功能：汇总公共分支置零后的解码覆盖率和新增可解码样本的 MSE。
# 输入：包含 loss_* 子目录的本地评测目录。
# 输出：public_zero_fill_summary.csv、public_zero_fill_cases.csv。
# ======================================================================

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def summarize(root):
    """读取逐 GOP CSV，按目标丢包率归并三个种子的结果。"""
    grouped = defaultdict(list)
    for path in sorted(root.glob("loss_*/gop_metrics.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[float(row["target_loss_fraction"])].append(row)
    if not grouped:
        raise ValueError(f"找不到逐 GOP 指标：{root}")

    summaries, cases = [], []
    for rate, rows in sorted(grouped.items()):
        failed = sum(row["decoder_ready"] == "False" for row in rows)
        failed_public = [r for r in rows if r["receiver_status"] == "failed_public_incomplete"]
        fallback = [r for r in failed_public if r["public_zero_fill_mse"]]
        # 误差只比较新增可解码的 GOP 与各自的无丢失 int8 结果。
        mse = [float(r["public_zero_fill_mse"]) for r in fallback]
        delta = [float(r["public_zero_fill_delta_mse"]) for r in fallback]
        summaries.append({
            "target_loss_fraction": rate,
            "gop_trials": len(rows),
            "baseline_failed_gops": failed,
            "public_incomplete_gops": len(failed_public),
            "newly_decoded_gops": len(fallback),
            "failed_gops_after_fallback": failed - len(fallback),
            "failed_fraction_after_fallback": (failed - len(fallback)) / len(rows),
            "paired_no_loss_mse": mean(float(r["no_loss_int8_mse"]) for r in fallback) if fallback else None,
            "mean_fallback_mse": mean(mse) if mse else None,
            "median_fallback_mse": median(mse) if mse else None,
            "max_fallback_mse": max(mse) if mse else None,
            "mean_fallback_delta_mse": mean(delta) if delta else None,
        })
        cases.extend(failed_public)
    return summaries, cases


def main():
    """保存分档汇总以及所有公共分支不完整的样本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries, cases = summarize(args.root)
    for name, rows in (("public_zero_fill_summary.csv", summaries),
                       ("public_zero_fill_cases.csv", cases)):
        if rows:
            with (args.root / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
