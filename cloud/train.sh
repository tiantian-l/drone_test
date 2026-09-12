#!/usr/bin/env bash
# Unified GPU training launcher. Run after cloud/setup.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"
CONDA_INIT="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [ ! -f "${CONDA_INIT}" ]; then
  echo "Conda not found at ${CONDA_ROOT}. Run 'bash cloud/setup.sh' first or set CONDA_ROOT." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_INIT}"
conda activate "${ENV_NAME}"
PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
if [ ! -x "${PY}" ]; then
  echo "Environment '${ENV_NAME}' not found. Run 'bash cloud/setup.sh' first or set ENV_NAME." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PATH="${CONDA_ROOT}/envs/${ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"
LOGDIR="${LOGDIR:-$HOME/logdir/drone_nav/{timestamp}}"
TRAIN_RATIO="${TRAIN_RATIO:-1024}"
ENVS="${ENVS:-8}"

cd "${REPO_ROOT}/third_party/dreamerv3"
"${PY}" dreamerv3/main.py \
  --configs drone_nav \
  --logdir "${LOGDIR}" \
  --jax.platform cuda \
  --run.envs "${ENVS}" \
  --run.train_ratio "${TRAIN_RATIO}" \
  --logger.outputs 'jsonl,scope,tensorboard' \
  --env.drone.log_image True \
  "$@"
