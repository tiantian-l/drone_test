#!/usr/bin/env python3
"""Compare the post-fork evaluation tail of A-E plateau diagnostics."""

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "success", "collision", "crash", "timeout", "final_distance",
    "min_distance", "min_lidar_dist", "time_to_goal")
LABELS = {
    "a_control": "A Control",
    "b_entropy": "B Entropy",
    "c_horizon": "C Horizon",
    "d_low_lr": "D Low-LR",
    "e_replay_1m": "E Replay-1M",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_evals(path, fork_step):
    by_step = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            step = row.get("step", row.get("_step"))
            if not finite(step) or int(step) < fork_step:
                continue
            values = {}
            for metric in METRICS:
                key = f"eval_epstats/log/{metric}"
                if finite(row.get(key)):
                    values[metric] = float(row[key])
            if values:
                by_step[int(step)] = values
    return sorted(by_step.items())


def mean(values):
    values = [value for value in values if finite(value)]
    return statistics.fmean(values) if values else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fork_root", type=Path)
    parser.add_argument("--fork-step", required=True, type=int)
    parser.add_argument("--tail", type=int, default=3)
    args = parser.parse_args()
    if args.tail < 1:
        raise SystemExit("--tail must be at least 1")

    results = {}
    for branch, label in LABELS.items():
        path = args.fork_root / branch / "metrics.jsonl"
        if not path.is_file():
            continue
        curve = read_evals(path, args.fork_step)
        tail = curve[-args.tail:]
        if tail:
            results[branch] = {
                "label": label,
                "evals": len(curve),
                "last_step": tail[-1][0],
                **{
                    metric: mean([values.get(metric) for _, values in tail])
                    for metric in METRICS
                },
            }

    if not results:
        raise SystemExit(f"No post-fork eval metrics found below {args.fork_root}")
    baseline = results.get("a_control", {}).get("success", math.nan)
    print("| Branch | Evals | Last step | Success tail mean | vs A | Collision | Crash | Timeout | Final dist |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for branch in LABELS:
        if branch not in results:
            continue
        row = results[branch]
        delta = row["success"] - baseline
        delta_text = f"{delta:+.4f}" if finite(delta) else "n/a"
        value = lambda key: f"{row[key]:.4f}" if finite(row[key]) else "n/a"
        print(
            f"| {row['label']} | {row['evals']} | {row['last_step']} | "
            f"{value('success')} | {delta_text} | {value('collision')} | "
            f"{value('crash')} | {value('timeout')} | "
            f"{value('final_distance')} |")


if __name__ == "__main__":
    main()
