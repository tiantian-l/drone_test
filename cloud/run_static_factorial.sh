#!/usr/bin/env bash
# Train the RQ1/RQ2 2x2 static factorial on a Linux GPU host.
# Run after the unified Conda setup: cloud/setup.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"
VENV_PY="${REPO_ROOT}/.venv-gpu/bin/python"
CONDA_PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"

if [[ -x "${CONDA_PY}" ]]; then
  PY="${CONDA_PY}"
  RUNTIME_BIN="${CONDA_ROOT}/envs/${ENV_NAME}/bin"
elif [[ -x "${VENV_PY}" ]]; then
  PY="${VENV_PY}"
  RUNTIME_BIN="${REPO_ROOT}/.venv-gpu/bin"
else
  echo "No training Python found." >&2
  echo "Run cloud/setup.sh first." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export PATH="${RUNTIME_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"

conditions_csv="${CONDITIONS:-10_sparse,10_dense,20_sparse,20_dense}"
seeds_csv="${SEEDS:-0}"
log_root="${LOG_ROOT:-$HOME/logdir/drone_static_factorial}"
jax_platform="${JAX_PLATFORM:-cuda}"
dry_run="${DRY_RUN:-0}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
main_py="${REPO_ROOT}/third_party/dreamerv3/dreamerv3/main.py"

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

IFS=',' read -r -a conditions <<< "${conditions_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

for condition in "${conditions[@]}"; do
  case "${condition}" in
    10_sparse|10_dense|20_sparse|20_dense) ;;
    *)
      echo "Unknown condition '${condition}'." >&2
      echo "Use 10_sparse,10_dense,20_sparse,20_dense." >&2
      exit 2
      ;;
  esac
  preset="drone_static_${condition}"
  for seed in "${seeds[@]}"; do
    logdir="${log_root}/${condition}/seed_${seed}"
    cmd=(
      "${PY}" "${main_py}"
      --configs drone_nav drone_perc_cnn_hi "${preset}" drone_reward_v2
      --seed "${seed}"
      --logdir "${logdir}"
      --jax.platform "${jax_platform}"
      --jax.compute_dtype "${compute_dtype}"
      --run.debug False
      --logger.outputs jsonl,scope,tensorboard
      --env.drone.log_image "${log_image}"
      --env.drone.video_every "${video_every}"
      "$@"
    )
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" != "1" ]]; then
      (
        cd "${REPO_ROOT}/third_party/dreamerv3"
        "${cmd[@]}"
      )
    fi
  done
done
