#!/bin/bash
# Submit from the repository root: sbatch cloud/train_static_10_sparse_slurm.sh
#SBATCH --job-name=10_sparse_bs32_s0
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
  echo "Submit from the repository root: sbatch cloud/train_static_10_sparse_slurm.sh" >&2
  exit 1
fi

export CONDA_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3
export ENV_NAME=drone
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# Use a separate directory per submission to avoid sharing checkpoints with
# the baseline or another job. Each submission starts a fresh training run.
experiment_root="/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_static_factorial_a100_bs32/job_${SLURM_JOB_ID:?Run via sbatch}"
echo "Run directory: ${experiment_root}/10_sparse/seed_0"
nvidia-smi

CONDITIONS=10_sparse \
SEEDS=0 \
LOG_ROOT="${experiment_root}" \
LOG_IMAGE=True \
VIDEO_EVERY=100 \
JAX_COMPUTE_DTYPE=bfloat16 \
bash cloud/run_static_factorial.sh \
  --batch_size 32 \
  --batch_length 64 \
  --run.train_ratio 512 \
  --run.steps 3e6 \
  --run.envs 16 \
  --run.eval_envs 16 \
  --run.debug False \
  --run.report_every 1800 \
  --env.drone.render_mode 3d
