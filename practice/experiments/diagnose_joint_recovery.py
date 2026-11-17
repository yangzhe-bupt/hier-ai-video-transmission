# ======================================================================
# 文件：diagnose_joint_recovery.py
# 功能：从已保存的包事件还原缺失块，分析联合恢复相对公共补全的退化。
# 输入：联合恢复结果目录。当前适配 1×4×24×32×32、16×16 区域分包。
# 输出：按样本的缺失统计、分组汇总和诊断报告。
# 运行：python practice/run.py experiments/diagnose_joint_recovery.py <结果目录>
# ======================================================================

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

# 统一入口保留虚拟文件位置，从项目根目录解析传输模块。
ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarize_joint_recovery import read_rows
from hier_video.transmission.feature_mapping import build_group_manifest


TOLERANCE = 1e-8


def missing_tiles(report, gop_id, groups):
    """从实际包状态得到公共缺块及各空间块的个体缺帧数。"""
    public_missing, individual_missing = set(), Counter()
    seen = set()
    for event in report["events"]:
        packet_gop, branch, group_id = event["packet_id"].split(":")
        group_id = int(group_id)
        group = groups[group_id]
        if int(packet_gop) != gop_id or branch != group.branch or group_id in seen:
            raise ValueError(f"包标识与映射不一致：{event['packet_id']}")
        seen.add(group_id)
        if event["status"] == "received":
            continue
        region = group.tensor_region
        tile = (region.h, region.w)
        if branch == "public":
            public_missing.add(tile)
        else:
            individual_missing[tile] += 1
    if seen != set(groups):
        raise ValueError(f"GOP {gop_id} 的包事件不完整")
    return public_missing, individual_missing


def diagnose(root):
    """将两路同时缺失的质量记录与同一次接收的包事件配对。"""
    manifest = build_group_manifest((1, 1, 24, 32, 32), (1, 4, 24, 32, 32))
    groups = {group.group_id: group for group in manifest.groups}
    traces, rows = {}, []
    for row in read_rows(root / "joint_recovery_cases.csv"):
        if int(row["fallback_individual_missing_elements"]) == 0:
            continue
        key = (row["target_loss_fraction"], row["seed"])
        if key not in traces:
            rate, seed = key
            path = root / f"loss_{rate}_seed_{seed}" / "events.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            traces[key] = {gop["gop_id"]: gop for gop in document["gops"]}
        gop_id = int(row["gop_id"])
        trace = traces[key][gop_id]
        if trace["start_frame"] != int(row["start_frame"]):
            raise ValueError("包事件与质量记录的源视频帧号不一致")
        public, individual = missing_tiles(trace["packet_loss"], gop_id, groups)
        elements_per_tile = 24 * 16 * 16
        if (len(public) * elements_per_tile != int(row["missing_public_elements"])
                or sum(individual.values()) * elements_per_tile
                != int(row["fallback_individual_missing_elements"])):
            raise ValueError(f"GOP {gop_id} 的缺失元素数与接收审计不一致")

        gain = float(row["joint_gain_over_public"])
        outcome = "improved" if gain > TOLERANCE else (
            "worsened" if gain < -TOLERANCE else "unchanged"
        )
        rows.append({
            **row,
            "joint_gain_over_public": gain,
            "outcome": outcome,
            "public_missing_tiles": len(public),
            "individual_missing_tiles": len(individual),
            "max_missing_frames_per_tile": max(individual.values()),
            "overlapping_tiles": len(public & individual.keys()),
            "single_missing_frame_tiles": sum(n == 1 for n in individual.values()),
            "two_missing_frame_tiles": sum(n == 2 for n in individual.values()),
            "three_missing_frame_tiles": sum(n == 3 for n in individual.values()),
            "all_frames_missing_tiles": sum(n == 4 for n in individual.values()),
            "individual_missing_frames_by_tile": json.dumps(
                {f"{h[0]},{w[0]}": count for (h, w), count in sorted(individual.items())},
            ),
        })
    if not rows:
        raise ValueError("缺少两路同时缺失的配对样本")
    return rows


def main():
    """保存逐样本诊断和各类缺失模式的分母。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = diagnose(args.root)
    with (args.root / "joint_failure_diagnostics.csv").open(
        "w", encoding="utf-8-sig", newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 联合恢复退化诊断",
        "",
        "使用已有包事件和配对 MSE。仅统计公共、个体都缺失且两路反量化参数齐全的样本。"
        "同一片段可在不同种子或目标丢包率下重复出现。",
        "",
        "|分组|样本数|改善|持平|退化|平均 MSE 降幅|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field in ("max_missing_frames_per_tile", "overlapping_tiles"):
        by_value = defaultdict(list)
        for row in rows:
            by_value[row[field]].append(row)
        label = "单块最大缺帧数" if field.startswith("max") else "公共与个体缺块重叠数"
        for value, selected in sorted(by_value.items()):
            outcomes = Counter(row["outcome"] for row in selected)
            gain = mean(row["joint_gain_over_public"] for row in selected)
            lines.append(
                f"|{label}={value}|{len(selected)}|{outcomes['improved']}|"
                f"{outcomes['unchanged']}|{outcomes['worsened']}|{gain:.8f}|"
            )
    lines.extend(["", "下表列出退化样本。MSE 上升量为联合恢复减去仅公共补全。", "",
                  "|起始帧|种子|目标丢包率|单块最大缺帧|重叠块数|MSE 上升量|",
                  "|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        if row["outcome"] == "worsened":
            rate = float(row["target_loss_fraction"])
            lines.append(
                f"|{row['start_frame']}|{row['seed']}|{rate:.0%}|"
                f"{row['max_missing_frames_per_tile']}|{row['overlapping_tiles']}|"
                f"{-row['joint_gain_over_public']:.9f}|"
            )
    (args.root / "DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
