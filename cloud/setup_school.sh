#!/usr/bin/env bash
# Setup on a SCHOOL Linux GPU machine for a *non-root* user (no apt / no sudo).
# Everything is installed under your home directory via a per-user Miniconda,
# so no system packages and no admin rights are needed. The NVIDIA driver is
# the only thing that must already be present (provided by the school); the
# CUDA runtime is shipped inside the JAX wheel.
#
# Usage (from the repo root):
#   bash cloud/setup_school.sh
#
# Override the install location if your home quota is small, e.g. point it at
# scratch storage:  CONDA_ROOT=/scratch/$USER/miniconda bash cloud/setup_school.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYVER=3.11
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"

# ---------------------------------------------------------------------------
# 1) Install Miniconda in user space (skip if already present)
# ---------------------------------------------------------------------------
if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
  echo "==> Installing Miniconda into ${CONDA_ROOT} (no root needed)"
  TMP_INSTALLER="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${TMP_INSTALLER}"
  bash "${TMP_INSTALLER}" -b -p "${CONDA_ROOT}"
  rm -f "${TMP_INSTALLER}"
else
  echo "==> Reusing existing Miniconda at ${CONDA_ROOT}"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2) Create the env with Python + ffmpeg (ffmpeg via conda, no apt needed)
# ---------------------------------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> Creating conda env '${ENV_NAME}' (python ${PYVER}, ffmpeg, git)"
  conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PYVER}" ffmpeg git
else
  echo "==> Reusing existing conda env '${ENV_NAME}'"
fi
conda activate "${ENV_NAME}"
pip install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# 3) Install the drone env + DreamerV3 (CUDA JAX bundles its own CUDA runtime)
# ---------------------------------------------------------------------------
echo "==> Installing gym-pybullet-drones (drone env only)"
pip install "numpy<2"
pip install -e "${REPO_ROOT}/third_party/gym-pybullet-drones"

echo "==> Installing DreamerV3 requirements with CUDA JAX"
grep -v '^jax' "${REPO_ROOT}/third_party/dreamerv3/requirements.txt" > /tmp/req-nojax.txt
pip install -r /tmp/req-nojax.txt
pip install "jax[cuda12]==0.4.33"

# ---------------------------------------------------------------------------
# 4) Sanity check
# ---------------------------------------------------------------------------
echo "==> Sanity check"
python - <<'PY'
import jax, gym_pybullet_drones, gymnasium, embodied, elements
print("jax devices:", jax.devices())
import drone_nav  # noqa
print("drone_nav import OK")
PY

echo "==> Setup complete."
echo "    Activate with: source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
