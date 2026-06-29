#!/usr/bin/env bash
# Launch GPU training on the school box (ordinary user, no sudo).
# Run from the repo root after cloud/setup_school_gpu.sh.
#   bash cloud/train_school_gpu.sh                 # default: drone_lidar preset
#   CONFIGS=drone_obstacles bash cloud/train_school_gpu.sh
#   ENVS=12 bash cloud/train_school_gpu.sh         # if the CPU has more cores
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-drone}"

# --- activate the same env that setup_school_gpu.sh created ------------------
if command -v conda >/dev/null 2>&1 && \
   conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
elif [ -f "${REPO_ROOT}/.venv-gpu/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv-gpu/bin/activate"
else
  echo "ERROR: no '${ENV_NAME}' conda env and no .venv-gpu found."
  echo "       Run 'bash cloud/setup_school_gpu.sh' first."
  exit 1
fi

# drone_nav lives at the repo root; expose it so `import drone_nav` resolves
# when main.py runs from inside third_party/dreamerv3.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Keep all artifacts under the repo (no /root, no shared system dirs).
CONFIGS="${CONFIGS:-drone_lidar}"
ENVS="${ENVS:-8}"
TRAIN_RATIO="${TRAIN_RATIO:-512}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/logs/${CONFIGS}/{timestamp}}"

# On shared GPUs, pin to one card and don't grab all the VRAM up front so you
# play nicely with other users. Override CUDA_VISIBLE_DEVICES to pick a card.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

cd "${REPO_ROOT}/third_party/dreamerv3"
python dreamerv3/main.py \
  --configs "${CONFIGS}" \
  --logdir "${LOGDIR}" \
  --jax.platform cuda \
  --run.envs "${ENVS}" \
  --run.train_ratio "${TRAIN_RATIO}" \
  --logger.outputs 'jsonl,scope,tensorboard' \
  --env.drone.log_image True \
  "$@"
