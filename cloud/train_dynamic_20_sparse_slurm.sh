#!/bin/bash
#SBATCH --job-name=20s_dyn10
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=normal-a100-80
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-dynamic-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-drone}"
export LOG_ROOT="${LOG_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_20_sparse_a100}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export JAX_COMPUTE_DTYPE=bfloat16
nvidia-smi
if [[ "${PREFLIGHT:-1}" == 1 ]]; then
  PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:${PYTHONPATH:-}" \
    "${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" cloud/check_dynamic_env.py
fi
bash cloud/run_dynamic_20_sparse.sh "$@"
