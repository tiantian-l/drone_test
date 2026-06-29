#!/usr/bin/env bash
# Launch GPU training on the SCHOOL machine (non-root user, conda env).
# Run from the repo root after cloud/setup_school.sh.
#   bash cloud/train_school.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# drone_nav lives at the repo root; expose it so `import drone_nav` resolves
# when main.py runs from inside third_party/dreamerv3.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Default logs to the user home (no shared/system paths). Override with LOGDIR.
LOGDIR="${LOGDIR:-$HOME/logdir/drone_nav/{timestamp}}"

cd "${REPO_ROOT}/third_party/dreamerv3"
python dreamerv3/main.py \
  --configs drone_nav \
  --logdir "${LOGDIR}" \
  --jax.platform cuda \
  --run.envs 8 \
  --run.train_ratio 1024 \
  --logger.outputs 'jsonl,scope,tensorboard' \
  --env.drone.log_image True \
  "$@"
