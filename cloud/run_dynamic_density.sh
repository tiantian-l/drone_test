#!/usr/bin/env bash
# Train the D-tier navigation task with reference-density moving cylinders.
# Run from anywhere after: bash cloud/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"

conda_init="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "${conda_init}" ]]; then
  echo "Conda initialization script not found: ${conda_init}" >&2
  echo "Run 'bash cloud/setup.sh' first or set CONDA_ROOT." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${conda_init}"
conda activate "${ENV_NAME}"

PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Conda environment Python not found: ${PY}" >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PATH="${CONDA_ROOT}/envs/${ENV_NAME}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"

seeds_csv="${SEEDS:-${SEED:-0}}"
steps="${STEPS:-5000000}"
log_root="${LOG_ROOT:-$HOME/logdir/drone_dynamic_density}"
jax_platform="${JAX_PLATFORM:-cuda}"
dry_run="${DRY_RUN:-0}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
main_py="${REPO_ROOT}/third_party/dreamerv3/dreamerv3/main.py"

# Older JAX/XLA builds need FP32 on Blackwell GPUs. Retain BF16 on earlier
# CUDA devices, matching the other school launchers.
compute_dtype="${JAX_COMPUTE_DTYPE:-auto}"
if [[ "${compute_dtype}" == "auto" ]]; then
  compute_dtype=float32
  if [[ "${jax_platform}" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    compute_major="${compute_cap%%.*}"
    if [[ "${compute_major}" =~ ^[0-9]+$ ]] && (( compute_major < 10 )); then
      compute_dtype=bfloat16
    fi
  fi
fi
if [[ "${compute_dtype}" != "float32" && "${compute_dtype}" != "bfloat16" ]]; then
  echo "JAX_COMPUTE_DTYPE must be auto, float32, or bfloat16" >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "${seeds_csv}"
for seed in "${seeds[@]}"; do
  logdir="${log_root}/seed_${seed}"
  cmd=(
    "${PY}" "${main_py}"
    --configs drone_nav drone_ablation_d drone_dynamic_density
    --seed "${seed}"
    --logdir "${logdir}"
    --run.steps "${steps}"
    --jax.platform "${jax_platform}"
    --jax.compute_dtype "${compute_dtype}"
    --logger.outputs jsonl,scope,tensorboard
    --env.drone.log_image "${log_image}"
    --env.drone.video_every "${video_every}"
    "$@"
  )

  echo "Dynamic D: seed=${seed}, steps=${steps}, logdir=${logdir}"
  echo "JAX compute dtype: ${compute_dtype}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${dry_run}" != "1" ]]; then
    (
      cd "${REPO_ROOT}/third_party/dreamerv3"
      "${cmd[@]}"
    )
  fi
done

echo "TensorBoard: tensorboard --logdir '${log_root}' --port 6006 --bind_all"
