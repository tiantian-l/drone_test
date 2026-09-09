#!/usr/bin/env bash
# Resume exactly one plateau diagnostic branch from the current full checkpoint.
# Required: LOGDIR=/path/to/20_sparse/seed_0 BRANCH=a_control|...
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

if [[ -z "${LOGDIR:-}" ]]; then
  echo "Set LOGDIR to the existing run directory containing ckpt/ and replay/." >&2
  exit 2
fi
if [[ -z "${BRANCH:-}" ]]; then
  echo "Set BRANCH to a_control, b_entropy, c_horizon, d_low_lr, or e_replay_1m." >&2
  exit 2
fi

logdir="${LOGDIR%/}"
for required in config.yaml ckpt replay eval_replay; do
  if [[ ! -e "${logdir}/${required}" ]]; then
    echo "Full-checkpoint component is missing: ${logdir}/${required}" >&2
    exit 2
  fi
done
for replay_dir in replay eval_replay; do
  if ! compgen -G "${logdir}/${replay_dir}/*.npz" >/dev/null; then
    echo "No replay chunks found in: ${logdir}/${replay_dir}" >&2
    exit 2
  fi
done

case "${BRANCH}" in
  a_control) preset=drone_plateau_a_control ;;
  b_entropy) preset=drone_plateau_b_entropy ;;
  c_horizon) preset=drone_plateau_c_horizon ;;
  d_low_lr) preset=drone_plateau_d_low_lr ;;
  e_replay_1m) preset=drone_plateau_e_replay_1m ;;
  *)
    echo "Unknown BRANCH '${BRANCH}'." >&2
    echo "Use a_control, b_entropy, c_horizon, d_low_lr, or e_replay_1m." >&2
    exit 2
    ;;
esac

export PYTHONNOUSERSITE=1
export PATH="${RUNTIME_BIN}:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"

# Supports both flat ckpt/ and timestamped ckpt/<generation>/ layouts.
checkpoint_step="$("${PY}" "${REPO_ROOT}/cloud/checkpoint_step.py" "${logdir}/ckpt")"
extra_steps="${EXTRA_STEPS:-500000}"
if [[ -n "${TARGET_STEPS:-}" ]]; then
  if [[ ! "${TARGET_STEPS}" =~ ^[1-9][0-9]*$ ]] || (( TARGET_STEPS <= checkpoint_step )); then
    echo "TARGET_STEPS must be an integer greater than checkpoint step ${checkpoint_step}." >&2
    exit 2
  fi
  target_steps="${TARGET_STEPS}"
else
  if [[ ! "${extra_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EXTRA_STEPS must be a positive integer, got '${extra_steps}'." >&2
    exit 2
  fi
  target_steps=$((checkpoint_step + extra_steps))
fi

task_preset="${TASK_PRESET:-drone_static_20_sparse}"
seed="${SEED:-0}"
jax_platform="${JAX_PLATFORM:-cuda}"
compute_dtype="${JAX_COMPUTE_DTYPE:-auto}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
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

cmd=(
  "${PY}" "${main_py}"
  --configs drone_nav drone_perc_cnn_hi "${task_preset}" drone_reward_v2 "${preset}"
  --seed "${seed}"
  --logdir "${logdir}"
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

echo "Branch:                 ${BRANCH} (${preset})"
echo "Run directory:          ${logdir}"
echo "Source checkpoint step: ${checkpoint_step}"
echo "Target total step:      ${target_steps}"
printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${dry_run}" != "1" ]]; then
  cd "${REPO_ROOT}/third_party/dreamerv3"
  "${cmd[@]}"
fi
