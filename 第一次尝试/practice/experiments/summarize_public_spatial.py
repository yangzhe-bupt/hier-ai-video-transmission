# ======================================================================
# 文件：summarize_public_spatial.py
# 功能：核对公共补全与置零的逐 GOP 条件，保存配对误差与样例。
# 输入：置零输出目录、空间补全输出目录。
# 输出：分档汇总 CSV、逐样本 CSV、人可读报告。
# ======================================================================
import argparse
import csv
from pathlib import Path
from statistics import mean


def read_rows(path):
    """读取逐 GOP 指标，保留样本顺序。"""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(baseline, output):
    """核对 12 个实验条件，汇总同一 GOP 的三种重建误差。"""
    summaries, cases = [], []
    for rate in ("0.0", "0.1", "0.2", "0.3"):
        rows = []
        for seed in (314, 315, 316):
            case = f"loss_{rate}_seed_{seed}"
            before, after = baseline / case, output / case
            if (before / "events.json").read_bytes() != (after / "events.json").read_bytes():
                raise ValueError(f"{case} 包事件不一致")
            reference = read_rows(before / "gop_metrics.csv")
            candidate = read_rows(after / "gop_metrics.csv")
            if len(reference) != 12 or len(candidate) != 12:
                raise ValueError(f"{case} 预期有 12 个 GOP")
            for old, new in zip(reference, candidate):
                if any(new[k] != v for k, v in old.items()):
                    raise ValueError(f"{case} GOP {old['gop_id']} 置零指标不同")
                if bool(new["public_zero_fill_mse"]) != bool(new["public_spatial_mse"]):
                    raise ValueError(f"{case} GOP {old['gop_id']} 两种方法的覆盖范围不同")
                rows.append(new)
        paired = [row for row in rows if row["public_spatial_mse"]]
        gains = [float(row["public_spatial_mse_reduction"]) for row in paired]
        cases.extend(paired)
        summaries.append({
            "target_loss_fraction": rate,
            "gop_trials": len(rows),
            "failed_before_fallback": sum(row["decoder_ready"] == "False" for row in rows),
            "paired_gops": len(paired),
            "improved": sum(gain > 1e-8 for gain in gains),
            "unchanged": sum(abs(gain) <= 1e-8 for gain in gains),
            "worsened": sum(gain < -1e-8 for gain in gains),
            "paired_no_loss_mse": mean(float(r["no_loss_int8_mse"]) for r in paired) if paired else None,
            "paired_zero_mse": mean(float(r["public_zero_fill_mse"]) for r in paired) if paired else None,
            "paired_spatial_mse": mean(float(r["public_spatial_mse"]) for r in paired) if paired else None,
        })
    return summaries, sorted(cases, key=lambda row: float(row["public_spatial_mse_reduction"]))


def main():
    """写出分档统计、按退化程度排序的样本及报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summaries, cases = summarize(args.baseline, args.output)
    for name, rows in (("public_spatial_summary.csv", summaries), ("public_spatial_cases.csv", cases)):
        if rows:
            with (args.output / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = [
        "# 公共分支空间补全对照",
        "",
        "第314轮权重、Sintel 的12个固定GOP和三个种子。12组包事件逐字节一致，置零指标逐字段一致。"
        "两种方法使用同一接收结果，个体残差保持接收端的置零策略。",
        "",
        "|目标丢包率|配对样本|改善 / 持平 / 退化|无丢失 MSE|公共置零 MSE|空间补全 MSE|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        rate = f"{float(row['target_loss_fraction']):.0%}"
        count = row["paired_gops"]
        counts = f"{row['improved']} / {row['unchanged']} / {row['worsened']}"
        metrics = [f"{row[k]:.8f}" if row[k] is not None else "—"
                   for k in ("paired_no_loss_mse", "paired_zero_mse", "paired_spatial_mse")]
        lines.append(f"|{rate}|{count}|{counts}|" + "|".join(metrics) + "|")
    lines.extend([
        "", "MSE 差值绝对值不超过 1e-8 记为持平。"
        "质量汇总涵盖公共分支有缺块、两支量化参数均可取得的样本。"
        "public_spatial_cases.csv 按误差改善量升序排序，负值表示退化。", "",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    for row in summaries:
        print(row)
    for row in cases[:3]:
        print({k: row[k] for k in ("start_frame", "seed", "target_loss_fraction",
                                  "public_zero_fill_mse", "public_spatial_mse")})


if __name__ == "__main__":
    main()
