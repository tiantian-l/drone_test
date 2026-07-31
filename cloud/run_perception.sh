#!/usr/bin/env bash
# Launch the perception-encoding ablation on the SCHOOL machine (non-root Conda
# env). Run from the repository root after: bash cloud/setup_school.sh
#
# Four groups, identical task/optimizer/step-budget, differing only in how the
# LiDAR is encoded / its resolution:
#   flat    -> state + flattened LiDAR through one shared MLP, 10 deg x 4 = 144.
#   cnn     -> state through the MLP, LiDAR through a dedicated 1D circular
#              convolution branch, same 10 deg x 4 = 144 as the baseline.
#   cnn_az  -> split CNN encoding, azimuth-only densification 5 deg x 4 = 288.
#   cnn_hi  -> split CNN encoding, both axes densified 5 deg x 8 = 576.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"

conda_init="${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -f "${conda_init}" ]]; then
  echo "Conda initialization script not found: ${conda_init}" >&2
  echo "Run 'bash cloud/setup_school.sh' first or set CONDA_ROOT." >&2
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

variants_csv="${VARIANTS:-flat,cnn,cnn_az,cnn_hi}"
seeds_csv="${SEEDS:-0}"
# Optional post-ablation curriculum controls. Defaults preserve the original
# perception ablation. For Reward V2 use REWARD_CONFIG=drone_reward_v2 and set
# TASK_TIER to b first, then c.
task_tier="${TASK_TIER:-b}"
reward_config="${REWARD_CONFIG:-}"
log_root="${LOG_ROOT:-$HOME/logdir/drone_perception}"
jax_platform="${JAX_PLATFORM:-cuda}"
dry_run="${DRY_RUN:-0}"
log_image="${LOG_IMAGE:-True}"
video_every="${VIDEO_EVERY:-100}"
main_py="${REPO_ROOT}/third_party/dreamerv3/dreamerv3/main.py"

# JAX 0.4.33's older XLA/LLVM backend aborts while lowering some BF16->F16
# conversions for Blackwell (compute capability 10.x/12.x). Use FP32 there as a
# compatibility mode while retaining BF16 on older CUDA GPUs.
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

if [[ "${task_tier}" != "b" && "${task_tier}" != "c" ]]; then
  echo "TASK_TIER must be b or c" >&2
  exit 2
fi
if [[ -n "${reward_config}" && "${reward_config}" != "drone_reward_v2" ]]; then
  echo "REWARD_CONFIG must be empty or drone_reward_v2" >&2
  exit 2
fi

for variant in "${variants[@]}"; do
  case "${variant}" in
    flat)   preset=drone_perc_flat;    variant_dir=FLAT ;;
    cnn)    preset=drone_perc_cnn;     variant_dir=CNN ;;
    cnn_az) preset=drone_perc_cnn_az;  variant_dir=CNN_AZ ;;
    cnn_hi) preset=drone_perc_cnn_hi;  variant_dir=CNN_HI ;;
    *) echo "Unknown variant '${variant}'; use flat,cnn,cnn_az,cnn_hi" >&2; exit 2 ;;
  esac
  for seed in "${seeds[@]}"; do
    configs=(drone_nav "${preset}")
    if [[ "${task_tier}" == "c" ]]; then
      configs+=(drone_ablation_c)
    fi
    if [[ -n "${reward_config}" ]]; then
      configs+=("${reward_config}")
    fi
    if [[ "${task_tier}" == "b" && -z "${reward_config}" ]]; then
      # Preserve the original ablation log layout for the default invocation.
      logdir="${log_root}/${variant_dir}/seed_${seed}"
    else
      experiment_dir="${variant_dir}/task_${task_tier}"
      if [[ -n "${reward_config}" ]]; then
        experiment_dir+="/${reward_config}"
      fi
      logdir="${log_root}/${experiment_dir}/seed_${seed}"
    fi
    cmd=(
      "${PY}" "${main_py}"
      --configs "${configs[@]}"
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
