#!/bin/bash
# Submit from repository root: sbatch cloud/eval_10_dense_serial_slurm.sh
# Reuse the serial evaluator within this single GPU allocation.
#SBATCH --job-name=eval10d_serial
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-eval10d-serial-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
export RUN_DIR="${RUN_DIR:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_static_factorial_a100/10_dense/seed_0}"
exec bash cloud/eval_checkpoints_serial_slurm.sh
