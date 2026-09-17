#!/bin/bash
# Submit from repository root: sbatch cloud/eval_checkpoints_serial_slurm.sh
# Defaults to 10_sparse/seed_0; override RUN_DIR for another saved training run.
# Override CHECKPOINT_DIR for newer protocol-specific best_checkpoints folders.
# One process finishes and releases its GPU memory before the next starts.
#SBATCH --job-name=eval_serial
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-eval-serial-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
RUN_DIR="${RUN_DIR:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_static_factorial_a100/10_sparse/seed_0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/best_checkpoints}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_DIR/checkpoint_evaluations}"
EVAL_MAPS="${EVAL_MAPS:-128}"
EVAL_REPEATS="${EVAL_REPEATS:-1}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
PYTHON="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
test -f "$RUN_DIR/config.yaml"
test -x "$PYTHON"
if [[ ! "$EVAL_REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$EVAL_MAPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_MAPS must be a positive integer" >&2
  exit 2
fi
shopt -s nullglob
checkpoints=("$CHECKPOINT_DIR"/step_*.ckpt)
if (( ${#checkpoints[@]} == 0 )); then
  echo "No step_*.ckpt checkpoints found in $CHECKPOINT_DIR" >&2
  exit 2
fi
batch_dir="$EVAL_ROOT/job_${SLURM_JOB_ID}_${EVAL_MAPS}maps_${EVAL_DTYPE}"
echo "Evaluating ${#checkpoints[@]} checkpoints serially; results: $batch_dir"
nvidia-smi
for checkpoint in "${checkpoints[@]}"; do
  name="${checkpoint##*/}"
  name="${name%.ckpt}"
  for ((repeat=1; repeat<=EVAL_REPEATS; repeat++)); do
    echo "Evaluating $name (repeat $repeat/$EVAL_REPEATS)"
    "$PYTHON" cloud/eval_checkpoint.py \
      --config "$RUN_DIR/config.yaml" \
      --checkpoint "$checkpoint" \
      --dtype "$EVAL_DTYPE" \
      --eval-maps "$EVAL_MAPS" \
      --output "$batch_dir/$name/repeat_$repeat"
  done
done
echo "All checkpoint evaluations completed: $batch_dir"
