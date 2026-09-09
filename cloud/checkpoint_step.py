#!/usr/bin/env python3
"""Print the integer step stored in an Elements checkpoint step.pkl."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    path = args.checkpoint
    if path.is_dir():
        path = path / "step.pkl"
    if not path.is_file():
        raise SystemExit(f"Checkpoint step file not found: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    print(unwrap_step(value))


if __name__ == "__main__":
    main()
