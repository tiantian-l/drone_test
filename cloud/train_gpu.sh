#!/usr/bin/env bash
# Launch GPU training. Run from the repo root after cloud/setup_gpu.sh.
#   bash cloud/train_gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv-gpu/bin/activate"

# drone_nav lives at the repo root; expose it so `import drone_nav` resolves
# when main.py runs from inside third_party/dreamerv3.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOGDIR="${LOGDIR:-/root/autodl-tmp/logdir/drone_obstacles/{timestamp}}"

cd "${REPO_ROOT}/third_party/dreamerv3"
python dreamerv3/main.py \
  --configs drone_obstacles \
  --logdir "${LOGDIR}" \
  --jax.platform cuda \
  --run.envs 8 \
  --run.train_ratio 1024 \
  --logger.outputs 'jsonl,scope,tensorboard' \
  --env.drone.log_image True \
  "$@"
