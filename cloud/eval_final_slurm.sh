#!/bin/bash
# sbatch cloud/eval_final_slurm.sh SCENARIO RUN_DIR CHECKPOINT [OUTPUT_DIR]
#SBATCH --job-name=final_eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-final-eval-%j.out
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: sbatch cloud/eval_final_slurm.sh SCENARIO RUN_DIR CHECKPOINT [OUTPUT_DIR]" >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
scenario="$1"
case "$scenario" in
  10_sparse|10_dense|20_sparse|20_dense|20_sparse_dynamic) ;;
  *) echo "Unknown test scenario: $scenario" >&2; exit 2 ;;
esac
run_dir="${2%/}"
checkpoint="${3%/}"
# Infer the source label from RUN_DIR/<seed>; override for custom layouts.
source_parent="$(dirname "$run_dir")"
policy_label="${POLICY_LABEL:-$(basename "$source_parent")_$(basename "$run_dir")}"
if [[ ! "$policy_label" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "POLICY_LABEL must contain only letters, digits, underscores, dots, or hyphens, starting with a letter or digit." >&2
  exit 2
fi
output="${4:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval/${scenario}/job_${SLURM_JOB_ID:?Run via sbatch}}"
echo "Evaluation scenario: $scenario; source policy: $policy_label"
echo "Checkpoint: $checkpoint"
echo "Results: $output"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:$PWD/third_party/gym-pybullet-drones:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
manifest="${TEST_SUITE_DIR:-$PWD/cloud/test_maps_v1}/${scenario}.json"
test -f "$run_dir/config.yaml"
test -f "$manifest"
test -e "$checkpoint"
nvidia-smi
exec "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" cloud/eval_final.py \
  --config "$run_dir/config.yaml" --checkpoint "$checkpoint" \
  --policy-label "$policy_label" \
  --manifest "$manifest" --output "$output" --dtype "${EVAL_DTYPE:-bfloat16}"
