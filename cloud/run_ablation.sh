#!/usr/bin/env bash
# Launch A-D ablation training on the SCHOOL machine (non-root Conda env).
# Run from the repository root after: bash cloud/setup.sh
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

# Match cloud/train.sh: activate the named environment and always use
# that environment's interpreter instead of a system/user Python.
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

variants_csv="${VARIANTS:-a,b,c,d}"
seeds_csv="${SEEDS:-0}"
log_root="${LOG_ROOT:-$HOME/logdir/drone_ablation}"
jax_platform="${JAX_PLATFORM:-cuda}"
dry_run="${DRY_RUN:-0}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
main_py="${REPO_ROOT}/third_party/dreamerv3/dreamerv3/main.py"

# JAX 0.4.33's older XLA/LLVM backend aborts while lowering some BF16->F16
# conversions for Blackwell (compute capability 10.x/12.x). Use FP32 there as
# a compatibility mode while retaining BF16 on older CUDA GPUs. Users can force
# either mode with JAX_COMPUTE_DTYPE=float32 or bfloat16.
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
echo "JAX compute dtype: ${compute_dtype}"

IFS=',' read -r -a variants <<< "${variants_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

for variant in "${variants[@]}"; do
  case "${variant}" in
    a|A) variant=a ;;
    b|B) variant=b ;;
    c|C) variant=c ;;
    d|D) variant=d ;;
    *) echo "Unknown variant '${variant}'; use a,b,c,d" >&2; exit 2 ;;
  esac
  for seed in "${seeds[@]}"; do
    case "${variant}" in
      a) variant_dir=A ;;
      b) variant_dir=B ;;
      c) variant_dir=C ;;
      d) variant_dir=D ;;
    esac
    logdir="${log_root}/${variant_dir}/seed_${seed}"
    cmd=(
      "${PY}" "${main_py}"
      --configs drone_nav "drone_ablation_${variant}"
      --seed "${seed}"
      --logdir "${logdir}"
      --jax.platform "${jax_platform}"
      --jax.compute_dtype "${compute_dtype}"
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
