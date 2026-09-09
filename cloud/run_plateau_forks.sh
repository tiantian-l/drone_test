#!/usr/bin/env bash
# Resume A-E diagnostic branches from independent copies of one full checkpoint.
# Required: SOURCE_LOGDIR=/path/to/20_sparse/seed_0
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
  echo "No training Python found. Run cloud/setup_school.sh or cloud/setup_gpu.sh first." >&2
  exit 1
fi

if [[ -z "${SOURCE_LOGDIR:-}" ]]; then
  echo "Set SOURCE_LOGDIR to the completed run directory containing ckpt/ and replay/." >&2
  exit 2
fi

source_logdir="${SOURCE_LOGDIR%/}"
for required in ckpt/step.pkl ckpt/agent.pkl replay eval_replay; do
  if [[ ! -e "${source_logdir}/${required}" ]]; then
    echo "Full-checkpoint component is missing: ${source_logdir}/${required}" >&2
    exit 2
  fi
done

export PYTHONNOUSERSITE=1
export PATH="${RUNTIME_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"

checkpoint_step="$("${PY}" "${REPO_ROOT}/cloud/checkpoint_step.py" "${source_logdir}/ckpt")"
extra_steps="${EXTRA_STEPS:-500000}"
if [[ ! "${extra_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXTRA_STEPS must be a positive integer, got '${extra_steps}'." >&2
  exit 2
fi
target_steps=$((checkpoint_step + extra_steps))

fork_root="${FORK_ROOT:-${source_logdir}_plateau_forks}"
branches_csv="${BRANCHES:-a_control,b_entropy,c_horizon,d_low_lr,e_replay_1m}"
task_preset="${TASK_PRESET:-drone_static_20_sparse}"
seed="${SEED:-0}"
jax_platform="${JAX_PLATFORM:-cuda}"
compute_dtype="${JAX_COMPUTE_DTYPE:-auto}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
prepare_only="${PREPARE_ONLY:-0}"
dry_run="${DRY_RUN:-0}"
main_py="${REPO_ROOT}/third_party/dreamerv3/dreamerv3/main.py"

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
case "${compute_dtype}" in
  float32|bfloat16) ;;
  *) echo "JAX_COMPUTE_DTYPE must be auto, float32, or bfloat16." >&2; exit 2 ;;
esac

declare -A presets=(
  [a_control]=drone_plateau_a_control
  [b_entropy]=drone_plateau_b_entropy
  [c_horizon]=drone_plateau_c_horizon
  [d_low_lr]=drone_plateau_d_low_lr
  [e_replay_1m]=drone_plateau_e_replay_1m
)

IFS=',' read -r -a branches <<< "${branches_csv}"
if [[ "${dry_run}" != "1" ]]; then
  mkdir -p "${fork_root}"
fi

echo "Source checkpoint step: ${checkpoint_step}"
echo "Target total step:      ${target_steps}"
echo "Fork root:              ${fork_root}"

for branch in "${branches[@]}"; do
  preset="${presets[${branch}]:-}"
  if [[ -z "${preset}" ]]; then
    echo "Unknown branch '${branch}'. Use: ${!presets[*]}" >&2
    exit 2
  fi
  branch_logdir="${fork_root}/${branch}"

  if [[ "${dry_run}" != "1" ]]; then
    if [[ -e "${branch_logdir}" ]]; then
      echo "Refusing to overwrite existing branch directory: ${branch_logdir}" >&2
      exit 2
    fi
    mkdir -p "${branch_logdir}"
    # GNU cp uses copy-on-write when the filesystem supports it and otherwise
    # falls back to a normal independent copy. Never share writable replay dirs.
    cp -a --reflink=auto "${source_logdir}/." "${branch_logdir}/"
  fi

  cmd=(
    "${PY}" "${main_py}"
    --configs drone_nav drone_perc_cnn_hi "${task_preset}" drone_reward_v2 "${preset}"
    --seed "${seed}"
    --logdir "${branch_logdir}"
    --jax.platform "${jax_platform}"
    --jax.compute_dtype "${compute_dtype}"
    --run.steps "${target_steps}"
    --run.eval_every_steps 1e5
    --run.debug False
    --logger.outputs jsonl,scope,tensorboard
    --env.drone.log_image "${log_image}"
    --env.drone.video_every "${video_every}"
    "$@"
  )

  printf 'Branch %s: ' "${branch}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${dry_run}" != "1" && "${prepare_only}" != "1" ]]; then
    (
      cd "${REPO_ROOT}/third_party/dreamerv3"
      "${cmd[@]}"
    )
  fi
done
