#!/usr/bin/env python3
"""Summarize A-D DreamerV3 ablation metrics without third-party packages."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = {
    "success": ("eval_epstats/log/success", "eval_episode/log/success"),
    "collision": ("eval_epstats/log/collision", "eval_episode/log/collision"),
    "crash": ("eval_epstats/log/crash", "eval_episode/log/crash"),
    "timeout": ("eval_epstats/log/timeout", "eval_episode/log/timeout"),
    "final_distance": (
        "eval_epstats/log/final_distance", "eval_episode/log/final_distance"),
    "min_distance": (
        "eval_epstats/log/min_distance", "eval_episode/log/min_distance"),
}


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def read_curve(path):
    curve = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            step = row.get("step", row.get("_step"))
            values = {}
            for name, keys in METRICS.items():
                for key in keys:
                    if finite_number(row.get(key)):
                        values[name] = float(row[key])
                        break
            if finite_number(step) and values:
                curve.append((int(step), values))
    return curve


def summarize_run(path, threshold):
    curve = read_curve(path)
    if not curve:
        return None
    final_step, final = curve[-1]
    threshold_step = next(
        (step for step, values in curve if values.get("success", -1) >= threshold),
        None)
    return {"step": final_step, "threshold_step": threshold_step, **final}


def mean_std(values):
    values = [x for x in values if finite_number(x)]
    if not values:
        return ""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_root", nargs="?",
        default=str(Path.home() / "logdir" / "drone_ablation"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default=None,
                        help="CSV path (default: LOG_ROOT/summary.csv)")
    args = parser.parse_args()
    root = Path(args.log_root)
    runs = []
    for variant in "ABCD":
        for path in sorted((root / variant).glob("seed_*/metrics.jsonl")):
            summary = summarize_run(path, args.threshold)
            if summary:
                summary.update(variant=variant, seed=path.parent.name[5:])
                runs.append(summary)

    if not runs:
        raise SystemExit(f"No usable metrics.jsonl files found below {root}")

    fields = ["variant", "seed", "step", "success", "collision", "crash",
              "timeout", "final_distance", "min_distance", "threshold_step"]
    output = Path(args.output) if args.output else root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    print("| Variant | Seeds | Success | Collision | Crash | Timeout | "
          "Final distance | Steps to success threshold |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in "ABCD":
        group = [row for row in runs if row["variant"] == variant]
        if not group:
            continue
        def vals(key):
            return [row.get(key) for row in group]
        reached = [x for x in vals("threshold_step") if finite_number(x)]
        threshold = mean_std(reached) if reached else "not reached"
        print(f"| {variant} | {len(group)} | {mean_std(vals('success'))} | "
              f"{mean_std(vals('collision'))} | {mean_std(vals('crash'))} | "
              f"{mean_std(vals('timeout'))} | {mean_std(vals('final_distance'))} | "
              f"{threshold} |")
    print(f"\nPer-seed CSV: {output}")


if __name__ == "__main__":
    main()
