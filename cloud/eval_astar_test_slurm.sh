#!/bin/bash
# Frozen-test evaluation of the tuned Map-A* baseline (256 maps per scenario),
# using the same PID-reset manifests as the APF test run.
#   sbatch --array=0-4 cloud/eval_astar_test_slurm.sh [TUNING_ROOT] [OUTPUT_ROOT]
# Reads TUNING_ROOT/<scenario>/<scenario>_frozen_params.json. Run only after
# tuning has finished; never re-tune after looking at these results.
#SBATCH --job-name=test_astar
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-test-astar-%A_%a.out
set -euo pipefail
scenarios=(10_sparse 10_dense 20_sparse 20_dense 20_sparse_dynamic)
scenario="${scenarios[${SLURM_ARRAY_TASK_ID:?Submit with --array=0-4}]}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
tuning_root="${1:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/baseline_tuning/astar}"
out_root="${2:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/baseline_test/astar}"
params="$tuning_root/$scenario/${scenario}_frozen_params.json"
manifest="$PWD/cloud/test_maps_pidreset_v2/${scenario}.json"
test -f "$params" && test -f "$manifest"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/third_party/gym-pybullet-drones:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$out_root"
exec "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" -W ignore cloud/eval_astar.py \
  --manifest "$manifest" --params "$params" --output "$out_root/${scenario}_job_${SLURM_JOB_ID}"
