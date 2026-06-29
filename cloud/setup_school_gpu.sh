#!/usr/bin/env bash
# One-shot setup on a SHARED school GPU box where you are an ordinary user:
#   * NO sudo / NO apt   -> we never touch system packages.
#   * Python 3.11 + ffmpeg come from conda-forge (no root needed).
#   * jax[cuda12] pip wheels bundle the CUDA runtime, so only an NVIDIA driver
#     is required on the host -- no system CUDA toolkit / no sudo.
#
# Usage (on the school machine, inside the repo root):
#   bash cloud/setup_school_gpu.sh
#
# If conda is unavailable, the script falls back to an existing python3.11 +
# venv (you must already have a python3.11 on PATH or loaded via `module`).
set -euo pipefail

PYVER=3.11
# Reuse the SAME conda env as the simple A->B task: the obstacle/lidar work adds
# no new dependencies (pure NumPy + existing DreamerV3 stack). Override with
# ENV_NAME=... if you want a separate env.
ENV_NAME="${ENV_NAME:-drone}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# --- pick an installer: conda (preferred) or python venv (fallback) ----------
if command -v conda >/dev/null 2>&1; then
  USE_CONDA=1
  echo "==> Using conda to create env '${ENV_NAME}' (python ${PYVER} + ffmpeg)"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PYVER}" ffmpeg
  fi
  conda activate "${ENV_NAME}"
  PY=python
else
  USE_CONDA=0
  echo "==> conda not found; falling back to python${PYVER} -m venv"
  if ! command -v "python${PYVER}" >/dev/null 2>&1; then
    echo "ERROR: python${PYVER} not on PATH. Load it first, e.g. 'module load python/${PYVER}',"
    echo "       or install Miniconda in your home dir (no sudo needed):"
    echo "         wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "         bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    echo "         eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
    exit 1
  fi
  "python${PYVER}" -m venv .venv-gpu
  # shellcheck disable=SC1091
  source .venv-gpu/bin/activate
  PY=python
  echo "WARN: no conda -> ffmpeg may be missing (only needed for video export)."
fi

echo "==> Upgrading pip toolchain"
${PY} -m pip install -U pip setuptools wheel

echo "==> Installing gym-pybullet-drones (drone env only)"
# numpy<2 is required by DreamerV3; pin it first so the resolver is stable.
${PY} -m pip install "numpy<2"
${PY} -m pip install -e third_party/gym-pybullet-drones

echo "==> Installing DreamerV3 requirements with CUDA JAX"
# Replace the CPU JAX line with the CUDA wheel (bundles CUDA libs, no sudo).
grep -v '^jax' third_party/dreamerv3/requirements.txt > /tmp/req-nojax-school.txt
${PY} -m pip install -r /tmp/req-nojax-school.txt
${PY} -m pip install "jax[cuda12]==0.4.33"

echo "==> Sanity check"
${PY} - <<'PY'
import jax, gym_pybullet_drones, gymnasium, embodied, elements  # noqa
print("jax devices:", jax.devices())
import drone_nav  # noqa
print("drone_nav import OK")
PY

echo
if [ "${USE_CONDA}" = "1" ]; then
  echo "==> Done. Activate later with:  conda activate ${ENV_NAME}"
else
  echo "==> Done. Activate later with:  source ${REPO_ROOT}/.venv-gpu/bin/activate"
fi
echo "==> Then train with:  bash cloud/train_school_gpu.sh"
