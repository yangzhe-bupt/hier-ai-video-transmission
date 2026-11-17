# ======================================================================
# 文件：summarize_joint_recovery.py
# 功能：核对联合恢复与原公共补全的事件和指标，汇总四组配对结果。
# 输入：公共补全输出目录、联合恢复输出目录。
# 输出：分档汇总、逐样本记录和 REPORT.md。
# ======================================================================

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = (
    "no_loss_int8_mse",
    "public_zero_fill_mse",
    "public_spatial_mse",
    "fallback_residual_only_mse",
    "fallback_joint_mse",
)


def read_rows(path):
    """读取 UTF-8 CSV，保留 GOP 顺序。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(reference, output):
    """返回按丢失率汇总的统计和可降级解码的全部配对样本。"""
    grouped = defaultdict(list)
    for rate in ("0.0", "0.1", "0.2", "0.3"):
        for seed in (314, 315, 316):
            case = f"loss_{rate}_seed_{seed}"
            before, after = reference / case, output / case
            if (before / "events.json").read_bytes() != (after / "events.json").read_bytes():
                raise ValueError(f"{case} 包事件不同")
            baseline = read_rows(before / "gop_metrics.csv")
            candidate = read_rows(after / "gop_metrics.csv")
            if len(baseline) != 12 or len(candidate) != 12:
                raise ValueError(f"{case} GOP 数量不同")
            for old, new in zip(baseline, candidate):
                if any(new[key] != value for key, value in old.items()):
                    raise ValueError(f"{case} GOP {old['gop_id']} 基线指标不同")
                if bool(new["public_spatial_mse"]) != bool(new["fallback_joint_mse"]):
                    raise ValueError(f"{case} GOP {old['gop_id']} 联合重建覆盖率不同")
                grouped[rate].append(new)

    summary, cases = [], []
    for rate, rows in grouped.items():
        paired = [row for row in rows if row["fallback_joint_mse"]]
        for row in paired:
            # 正值表示联合比仅公共补全的 MSE 更低。
            row["joint_gain_over_public"] = (
                float(row["public_spatial_mse"]) - float(row["fallback_joint_mse"])
            )
        cases.extend(paired)
        # 同时给出全体公共缺块样本和公共、个体都缺失的子集。
        both_missing = [row for row in paired if int(row["fallback_individual_missing_elements"]) > 0]
        for subset, selected in (("public_missing", paired), ("both_missing", both_missing)):
            gains = [row["joint_gain_over_public"] for row in selected]
            summary.append({
                "target_loss_fraction": rate,
                "subset": subset,
                "paired_gops": len(selected),
                "improved": sum(gain > 1e-8 for gain in gains),
                "unchanged": sum(abs(gain) <= 1e-8 for gain in gains),
                "worsened": sum(gain < -1e-8 for gain in gains),
                **{key: mean(float(row[key]) for row in selected) if selected else None
                   for key in METRICS},
                "mean_joint_gain_over_public": mean(gains) if gains else None,
            })
    return summary, sorted(cases, key=lambda row: row["joint_gain_over_public"])


def main():
    """将分档结果和逐 GOP 样本写入现有评测目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary, cases = summarize(args.reference, args.output)
    for name, rows in (("joint_recovery_summary.csv", summary),
                       ("joint_recovery_cases.csv", cases)):
        if rows:
            with (args.output / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    lines = [
        "# 公共与个体残差联合恢复",
        "",
        "本地第314轮权重，12个固定GOP和三个种子。12组包事件逐字节与公共补全对照一致，原指标逐字段一致。",
        "四种方法共享同一接收结果。下表仅计公共分支缺块且两个分支的反量化参数齐全的GOP。",
        "",
        "|目标丢包率|配对 GOP|两路都置零|仅公共补全|仅残差恢复|联合恢复|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["subset"] != "public_missing":
            continue
        rate = float(row["target_loss_fraction"])
        metrics = [f"{row[key]:.8f}" if row[key] is not None else "—" for key in METRICS[1:]]
        lines.append(f"|{rate:.0%}|{row['paired_gops']}|" + "|".join(metrics) + "|")
    conflicting = [row for row in cases if int(row["fallback_individual_missing_elements"]) > 0]
    gains = [row["joint_gain_over_public"] for row in conflicting]
    if gains:
        better = sum(gain > 1e-8 for gain in gains)
        worse = sum(gain < -1e-8 for gain in gains)
        same = len(gains) - better - worse
        lines.extend([
            "", f"两路同时缺失的 {len(gains)} 次试验中，对比仅公共补全，联合恢复改善 {better} 次、"
            f"持平 {same} 次、退化 {worse} 次，平均 MSE 降低 {mean(gains):.8f}。",
            "MSE 差值绝对值不超过 1e-8 记为持平。逐样本 CSV 按联合恢复收益升序排列，负值为退化。",
        ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
