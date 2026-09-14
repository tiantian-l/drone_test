#!/bin/bash
# Submit from repository root with sbatch cloud/eval_20_sparse_slurm.sh
#SBATCH --job-name=eval20s
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-eval-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
RUN_DIR="${RUN_DIR:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_static_factorial_a100/20_sparse/seed_0}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best_checkpoints/step_0001400000_success_0.9531.ckpt}"
EVAL_ROOT="${EVAL_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/checkpoint_evaluations}"
test -f "$RUN_DIR/config.yaml"
test -e "$CHECKPOINT"
nvidia-smi
for repeat in 1 2; do
  "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" cloud/eval_checkpoint.py \
    --config "$RUN_DIR/config.yaml" --checkpoint "$CHECKPOINT" \
    --dtype "${EVAL_DTYPE:-bfloat16}" \
    --output "$EVAL_ROOT/job_${SLURM_JOB_ID}_${EVAL_DTYPE:-bfloat16}_repeat_${repeat}"
done
