#!/usr/bin/env bash
set -euo pipefail

# Feasibility runs for experiments C then D, one independent training seed.
# D receives the full five-million-step ceiling; runs resume from their own
# checkpoints when the same log directory is reused.
script_dir="$(cd "$(dirname "$0")" && pwd)"
seed="${SEED:-0}"
c_steps="${C_STEPS:-2500000}"
d_steps="${D_STEPS:-5000000}"

echo "[C] seed=${seed}, target_steps=${c_steps}"
VARIANTS=c SEEDS="${seed}" \
  bash "${script_dir}/run_ablation.sh" --run.steps "${c_steps}" "$@"

echo "[D] seed=${seed}, target_steps=${d_steps}"
VARIANTS=d SEEDS="${seed}" \
  bash "${script_dir}/run_ablation.sh" --run.steps "${d_steps}" "$@"
