#!/bin/bash
# Validation tuning of the Map-A* baseline, one array task per scenario.
#   sbatch --array=0-4 cloud/tune_astar_slurm.sh [OUTPUT_ROOT]
#   TASK_INFO=none sbatch --array=0-4 cloud/tune_astar_slurm.sh   # variant without flight region
# TASK_INFO=region (default): goal world position + flight region are given.
# TASK_INFO=none: observation only (goal-centred map, no region boundary).
# Array index: 0 10_sparse, 1 10_dense, 2 20_sparse, 3 20_dense, 4 20_sparse_dynamic.
# CPU-only work; the partition requires a GPU. Each task writes
# OUTPUT_ROOT/<scenario>/ and resumes from its episodes.jsonl if resubmitted.
# Summarize afterwards:  python cloud/report_baseline_tuning.py OUTPUT_ROOT
#SBATCH --job-name=tune_astar
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-tune-astar-%A_%a.out
set -euo pipefail
scenarios=(10_sparse 10_dense 20_sparse 20_dense 20_sparse_dynamic)
scenario="${scenarios[${SLURM_ARRAY_TASK_ID:?Submit with --array=0-4}]}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
task_info="${TASK_INFO:-region}"
case "$task_info" in region|none) ;; *) echo "TASK_INFO must be region or none" >&2; exit 2 ;; esac
suffix=""; [[ "$task_info" == none ]] && suffix="_noregion"
out_root="${1:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/baseline_tuning/astar${suffix}}"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/third_party/gym-pybullet-drones:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
if ! "$py" -c "import skimage" 2>/dev/null; then
  echo "scikit-image is missing. Install once on a login node: $py -m pip install scikit-image" >&2
  exit 2
fi
mkdir -p "$out_root"
echo "Tuning Map-A* (task_info=$task_info) on $scenario with ${SLURM_CPUS_PER_TASK} workers -> $out_root/$scenario"
exec "$py" -W ignore cloud/tune_astar.py --task-info "$task_info" --scenarios "$scenario" \
  --workers "${SLURM_CPUS_PER_TASK}" --output "$out_root/$scenario"
