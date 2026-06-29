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
  conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PYVER}" pip ffmpeg git
else
  echo "==> Reusing existing conda env '${ENV_NAME}'"
fi
conda activate "${ENV_NAME}"

# The school box already has a system Python 3.10 with packages in
# ~/.local; if conda activate fails to fully prepend the env (or a stray
# pip resolves to the system one), installs leak there and the env's
# python can never import them. Pin pip to the env interpreter and disable
# the per-user site so everything lands inside the conda env.
PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1
# The env may have been created without pip; bootstrap it if missing.
if ! "${PY}" -m pip --version >/dev/null 2>&1; then
  echo "==> pip missing in env; bootstrapping with ensurepip"
  "${PY}" -m ensurepip --upgrade || conda install -y -n "${ENV_NAME}" -c conda-forge pip
fi
PIP="${PY} -m pip --disable-pip-version-check"
echo "==> Using interpreter: ${PY}"
"${PY}" -c "import sys; print('site:', sys.prefix)"
${PIP} install -U pip setuptools wheel

# ---------------------------------------------------------------------------
# 3) Install the drone env + DreamerV3 (CUDA JAX bundles its own CUDA runtime)
# ---------------------------------------------------------------------------
echo "==> Installing gym-pybullet-drones (drone env only)"
${PIP} install "numpy<2"
${PIP} install -e "${REPO_ROOT}/third_party/gym-pybullet-drones"

echo "==> Installing DreamerV3 requirements with CUDA JAX"
# Drop jax/jaxlib (we pin 0.4.33 below) AND optax: the unpinned optax resolves
# to 0.2.8, which forces jax>=0.5.3 and silently breaks our 0.4.33 install.
# optax 0.2.4 is the newest release still compatible with jax 0.4.33.
grep -vE '^(jax|jaxlib|optax)' "${REPO_ROOT}/third_party/dreamerv3/requirements.txt" > /tmp/req-nojax.txt
${PIP} install -r /tmp/req-nojax.txt
${PIP} install "optax==0.2.4" "jax[cuda12]==0.4.33"

# ---------------------------------------------------------------------------
# 4) Sanity check (use the env's own interpreter, not whatever `python` is)
# ---------------------------------------------------------------------------
echo "==> Sanity check"
"${PY}" - <<'PY'
import jax, optax, gym_pybullet_drones, gymnasium, embodied, elements
print("jax:", jax.__version__, "optax:", optax.__version__)
print("jax devices:", jax.devices())
import drone_nav  # noqa
print("drone_nav import OK")
PY

echo "==> Setup complete."
echo "    Activate with: source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
