#!/usr/bin/env python3
"""Validate and summarize map-level deterministic evaluation results."""

import argparse
import collections
import json
from pathlib import Path


def load_cycles(path):
    cycles = collections.defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{lineno}: {exc}")
            cycles[int(row["eval_cycle"])].append(row)
    return cycles


def validate(rows, expected_maps, expected_workers):
    seeds = [int(row["map_seed"]) for row in rows]
    counts = collections.Counter(seeds)
    duplicates = sorted(seed for seed, count in counts.items() if count != 1)
    problems = []
    if len(rows) != expected_maps:
        problems.append(f"rows={len(rows)} (expected {expected_maps})")
    if len(counts) != expected_maps:
        problems.append(
            f"unique_maps={len(counts)} (expected {expected_maps})")
    if duplicates:
        problems.append(f"duplicate_maps={duplicates}")
    if expected_maps % expected_workers:
        problems.append(
            f"expected_maps={expected_maps} is not divisible by "
            f"expected_workers={expected_workers}")
        return problems
    quota = expected_maps // expected_workers
    workers = collections.Counter(int(row["worker"]) for row in rows)
    bad_workers = {
        worker: workers.get(worker, 0) for worker in range(expected_workers)
        if workers.get(worker, 0) != quota}
    if bad_workers:
        problems.append(f"per_worker_counts={bad_workers} (expected {quota})")
    slots = collections.defaultdict(set)
    for row in rows:
        slots[int(row["worker"])].add(int(row["map_slot"]))
    bad_slots = {
        worker: sorted(values) for worker, values in slots.items()
        if len(values) != quota}
    if bad_slots:
        problems.append(f"per_worker_slots={bad_slots} (expected {quota} unique)")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to eval_maps.jsonl")
    parser.add_argument("--expected-maps", type=int, default=64)
    parser.add_argument("--expected-workers", type=int, default=16)
    parser.add_argument(
        "--cycle", type=int, default=None,
        help="Eval cycle to inspect (default: latest cycle)")
    args = parser.parse_args()

    path = Path(args.path)
    cycles = load_cycles(path)
    if not cycles:
        raise SystemExit(f"No evaluation records found in {path}")
    cycle = max(cycles) if args.cycle is None else args.cycle
    if cycle not in cycles:
        raise SystemExit(f"Eval cycle {cycle} not found; available: {sorted(cycles)}")
    rows = cycles[cycle]
    problems = validate(rows, args.expected_maps, args.expected_workers)
    if problems:
        raise SystemExit("Invalid evaluation coverage: " + "; ".join(problems))

    outcomes = {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in ("success", "collision", "crash", "timeout")}
    step = rows[0].get("checkpoint_step", "unknown")
    print(f"Eval cycle: {cycle}  checkpoint step: {step}")
    print(f"Maps: {len(rows)} unique / {len(rows)} episodes")
    for key, value in outcomes.items():
        print(f"{key:>10}: {value:.4f} ({round(value * len(rows))}/{len(rows)})")

    failures = [row for row in rows if float(row.get("success", 0.0)) < 0.5]
    if failures:
        print("\nFailed maps:")
        print("map_seed,worker,slot,collision,crash,timeout,final_distance")
        for row in sorted(failures, key=lambda item: int(item["map_seed"])):
            print(
                f"{row['map_seed']},{row['worker']},{row['map_slot']},"
                f"{row['collision']:.0f},{row['crash']:.0f},"
                f"{row['timeout']:.0f},{row['final_distance']:.4f}")


if __name__ == "__main__":
    main()
