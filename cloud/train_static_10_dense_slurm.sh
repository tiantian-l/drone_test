#!/bin/bash
# New run: sbatch cloud/train_static_10_dense_slurm.sh
# Resume: sbatch cloud/train_static_10_dense_slurm.sh /absolute/path/10_dense/seed_N
#SBATCH --job-name=10_dense_bs32_s0
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
if [ ! -f cloud/run_static_factorial.sh ]; then
  echo "Submit from the repository root: sbatch cloud/train_static_10_dense_slurm.sh" >&2
  exit 1
fi

export CONDA_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3
export ENV_NAME=drone
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

if [[ $# -gt 1 ]]; then
  echo "Usage: sbatch cloud/train_static_10_dense_slurm.sh [/absolute/path/10_dense/seed_N]" >&2
  exit 2
fi
training_seed=0
if [[ $# -eq 1 ]]; then
  run_dir="${1%/}"
  if [[ ! "$run_dir" =~ ^(/.*)/10_dense/seed_([0-9]+)$ ]]; then
    echo "Resume directory must be an absolute path ending in /10_dense/seed_N." >&2
    exit 2
  fi
  experiment_root="${BASH_REMATCH[1]}"
  training_seed="${BASH_REMATCH[2]}"
  for component in config.yaml ckpt replay eval_replay; do
    if [[ ! -e "${run_dir}/${component}" ]]; then
      echo "Missing resume component: ${run_dir}/${component}" >&2
      exit 1
    fi
  done
  for replay_dir in replay eval_replay; do
    if ! compgen -G "${run_dir}/${replay_dir}/*.npz" >/dev/null; then
      echo "No replay chunks found: ${run_dir}/${replay_dir}" >&2
      exit 1
    fi
  done
  checkpoint_step="$("${CONDA_ROOT}/envs/${ENV_NAME}/bin/python" cloud/checkpoint_step.py "${run_dir}/ckpt")"
  if (( checkpoint_step >= 3000000 )); then
    echo "Checkpoint already reached the 3000000-step target: ${checkpoint_step}"
    exit 0
  fi
  echo "Resuming checkpoint step ${checkpoint_step}; target total steps: 3000000"
  # main.py rewrites config.yaml on startup.
  cp -n "${run_dir}/config.yaml" "${run_dir}/config.before_resume_${SLURM_JOB_ID:?Run via sbatch}.yaml"
else
  # Each new submission uses its own directory.
  experiment_root="/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/bs32/job_${SLURM_JOB_ID:?Run via sbatch}"
  run_dir="${experiment_root}/10_dense/seed_${training_seed}"
fi
echo "Run directory: ${run_dir}"
nvidia-smi

CONDITIONS=10_dense \
SEEDS="${training_seed}" \
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
