#!/usr/bin/env bash
# Same mixed environment as finetuning, with random initialization and fresh replay.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if (( $# )); then
  echo "Use MODE, SEED, STEPS, BATCH_SIZE and LOG_ROOT/LOGDIR; CLI overrides are disabled for this baseline." >&2
  exit 2
fi
export MODE="${MODE:-scratch}"
export SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
export STEPS="${STEPS:-3000000}"
export LOG_ROOT="${LOG_ROOT:-$HOME/logdir/drone_dynamic_scratch_a100}"
export LOG_IMAGE="${LOG_IMAGE:-True}"
export PYTHONUNBUFFERED=1
case "$MODE" in
  scratch)
    batch_size="${BATCH_SIZE:-32}"
    if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
      echo "BATCH_SIZE must be a positive integer." >&2
      exit 2
    fi
    # Do not inherit the static initialization checkpoint from a previous command.
    unset INIT_CHECKPOINT
    exec bash "$REPO_ROOT/cloud/run_dynamic_20_sparse.sh" \
      --batch_size "$batch_size" --run.from_checkpoint ''
    ;;
  resume)
    # Resume reads batch size and all task settings from this run's config.yaml.
    exec bash "$REPO_ROOT/cloud/run_dynamic_20_sparse.sh"
    ;;
  *) echo "This baseline supports MODE=scratch or MODE=resume only." >&2; exit 2 ;;
esac
