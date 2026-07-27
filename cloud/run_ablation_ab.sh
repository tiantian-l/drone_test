#!/usr/bin/env bash
set -euo pipefail

# School Conda runs for experiments A then B, one independent training seed.
# Defaults are deliberately shorter because these capabilities were already
# demonstrated previously. Override A_STEPS/B_STEPS when a longer run is needed.
script_dir="$(cd "$(dirname "$0")" && pwd)"
seed="${SEED:-0}"
a_steps="${A_STEPS:-500000}"
b_steps="${B_STEPS:-1000000}"

echo "[A] seed=${seed}, target_steps=${a_steps}"
VARIANTS=a SEEDS="${seed}" LOG_IMAGE="${LOG_IMAGE:-True}" \
  VIDEO_EVERY="${VIDEO_EVERY:-50}" \
  bash "${script_dir}/run_ablation.sh" --run.steps "${a_steps}" "$@"

echo "[B] seed=${seed}, target_steps=${b_steps}"
VARIANTS=b SEEDS="${seed}" LOG_IMAGE="${LOG_IMAGE:-True}" \
  VIDEO_EVERY="${VIDEO_EVERY:-50}" \
  bash "${script_dir}/run_ablation.sh" --run.steps "${b_steps}" "$@"
