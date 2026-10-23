#!/usr/bin/env bash
# Submit from the repository root; supply account/partition appropriate to your cluster.
#SBATCH --job-name=ppo_10_sparse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-ppo-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
bash cloud/run_ppo.sh "$@"
