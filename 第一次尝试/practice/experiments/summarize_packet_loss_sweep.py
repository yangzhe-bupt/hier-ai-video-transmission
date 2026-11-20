# ======================================================================
# 文件：summarize_packet_loss_sweep.py
# 功能：读取置零评测的逐条件汇总，合并同一目标丢失率下的多个种子。
# 输入：包含 loss_* 子目录的扫描输出目录。
# 输出：sweep_summary.csv，每个目标丢失率一行。
# ======================================================================

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize_sweep(root):
    """返回按目标丢失率排序的跨种子统计，保留各指标的原始分母。"""
    groups = defaultdict(list)
    for path in sorted(root.glob("loss_*/summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        groups[report["target_loss_fraction"]].append(report)
    if not groups:
        raise ValueError(f"找不到汇总结果：{root}")

    rows = []
    for rate, reports in sorted(groups.items()):
        seeds = sorted(report["seed"] for report in reports)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"丢失率 {rate} 有重复种子")
        gops = sum(report["gop_count"] for report in reports)
        failed = sum(report["failed_gop_count"] for report in reports)
        decoded = gops - failed
        sent = sum(report["sent_packets"] for report in reports)
        lost = sum(report["lost_packets"] for report in reports)
        row = {
            "target_loss_fraction": rate,
            "seeds": " ".join(map(str, seeds)),
            "gop_trials": gops,
            "decoded_gop_count": decoded,
            "failed_gop_count": failed,
            "failed_gop_fraction": failed / gops,
            "sent_packets": sent,
            "lost_packets": lost,
            "actual_packet_loss_fraction": lost / sent if sent else None,
            "sent_bytes": sum(report["sent_bytes"] for report in reports),
        }
        # 按成功解码的 GOP 数加权。全失败的种子仍计入失败率。
        for field in ("paired_no_loss_mse", "decoded_packet_loss_mse", "mean_delta_mse"):
            total = sum(report[field] * report["decoded_gop_count"]
                        for report in reports if report["decoded_gop_count"])
            row[field] = total / decoded if decoded else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="汇总本地包级置零实验")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = summarize_sweep(args.root)
    output = args.root / "sweep_summary.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output.resolve())


if __name__ == "__main__":
    main()
