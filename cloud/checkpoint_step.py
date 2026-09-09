#!/usr/bin/env python3
"""Print the step from the latest complete Elements checkpoint generation."""

import argparse
import pickle
from pathlib import Path


def unwrap_step(value):
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid checkpoint step")
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        scalar = value.item()
        if isinstance(scalar, int) and not isinstance(scalar, bool):
            return scalar
    if isinstance(value, dict):
        for key in ("step", "value", "count"):
            if key in value:
                return unwrap_step(value[key])
    raise TypeError(
        f"unsupported checkpoint step payload: {type(value).__name__}")


def complete_generation(path):
    return (
        path.is_dir()
        and (path / "done").is_file()
        and (path / "step.pkl").is_file()
        and (path / "agent.pkl").is_file()
    )


def resolve_generation(path):
    """Accept either one generation or its parent checkpoint directory."""
    if complete_generation(path):
        return path
    latest = path / "latest"
    if complete_generation(latest):
        return latest.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    generations = sorted(
        child for child in path.iterdir() if complete_generation(child))
    if not generations:
        raise FileNotFoundError(
            f"No complete checkpoint generation found below: {path}")
    return generations[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        generation = resolve_generation(args.checkpoint)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    with (generation / "step.pkl").open("rb") as handle:
        value = pickle.load(handle)
    print(unwrap_step(value))


if __name__ == "__main__":
    main()
