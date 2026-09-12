#!/bin/bash
# Submit from the repository root: sbatch cloud/train_static_10_sparse_slurm.sh
#SBATCH --job-name=drone_10_sparse_s0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
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

nvidia-smi

CONDITIONS=10_sparse \
SEEDS=0 \
LOG_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_static_factorial_a100 \
LOG_IMAGE=True \
VIDEO_EVERY=100 \
JAX_COMPUTE_DTYPE=bfloat16 \
bash cloud/run_static_factorial.sh \
  --run.steps 2e6 \
  --run.envs 16 \
  --run.eval_envs 16 \
  --run.debug False \
  --run.report_every 1800 \
  --env.drone.render_mode 3d
