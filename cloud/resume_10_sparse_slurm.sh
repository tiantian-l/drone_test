#!/bin/bash
# Submit from the repository root:
# sbatch cloud/resume_10_sparse_slurm.sh /absolute/path/10_sparse/seed_N [target_total_steps]
#SBATCH --job-name=10_sparse_bs32_resume
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
if [ ! -f cloud/run_static_factorial.sh ]; then
  echo "Submit from the repository root: sbatch cloud/resume_10_sparse_slurm.sh /absolute/path/10_sparse/seed_N [target_total_steps]" >&2
  exit 1
fi

export CONDA_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3
export ENV_NAME=drone
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Resume the existing run; the target is a total step count, not extra steps.
run_dir="${1:-}"
run_dir="${run_dir%/}"
target_steps="${2:-3000000}"
if [[ $# -lt 1 || $# -gt 2 || ! "$target_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: sbatch cloud/resume_10_sparse_slurm.sh /absolute/path/10_sparse/seed_N [target_total_steps as integer]" >&2
  exit 2
fi
if [[ ! "$run_dir" =~ ^(/.*)/10_sparse/seed_([0-9]+)$ ]]; then
  echo "Run directory must be an absolute path ending in /10_sparse/seed_N." >&2
  exit 2
fi
experiment_root="${BASH_REMATCH[1]}"
training_seed="${BASH_REMATCH[2]}"
for component in config.yaml ckpt replay eval_replay; do
  if [[ ! -e "${run_dir}/${component}" ]]; then
    echo "Missing resume component: ${run_dir}/${component}" >&2
    exit 1
  fi
done
for replay_dir in replay eval_replay; do
  if ! compgen -G "${run_dir}/${replay_dir}/*.npz" >/dev/null; then
    echo "No replay chunks found: ${run_dir}/${replay_dir}" >&2
    exit 1
  fi
done
checkpoint_step="$("${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" cloud/checkpoint_step.py "${run_dir}/ckpt")"
if (( target_steps <= checkpoint_step )); then
  echo "Checkpoint is already at ${checkpoint_step}; choose a larger total target than ${target_steps}." >&2
  exit 2
fi
echo "Resume directory: ${run_dir}"
echo "Checkpoint step: ${checkpoint_step}; target total steps: ${target_steps}"
# main.py rewrites config.yaml on startup; preserve the pre-resume configuration.
cp -n "${run_dir}/config.yaml" "${run_dir}/config.before_resume_${SLURM_JOB_ID:?Run via sbatch}.yaml"
nvidia-smi

CONDITIONS=10_sparse \
SEEDS="${training_seed}" \
LOG_ROOT="${experiment_root}" \
LOG_IMAGE=True \
VIDEO_EVERY=100 \
JAX_COMPUTE_DTYPE=bfloat16 \
bash cloud/run_static_factorial.sh \
  --batch_size 32 \
  --batch_length 64 \
  --run.train_ratio 512 \
  --run.steps "${target_steps}" \
  --run.envs 16 \
  --run.eval_envs 16 \
  --run.debug False \
  --run.report_every 1800 \
  --env.drone.render_mode 3d
