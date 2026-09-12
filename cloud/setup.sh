#!/usr/bin/env bash
# Unified, root-free setup for Linux GPU machines (school or cloud).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYVER="${PYVER:-3.11}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"
TMP_INSTALLER=""
REQ_FILE=""
cleanup() { [ -z "${TMP_INSTALLER}" ] || rm -f "${TMP_INSTALLER}"; [ -z "${REQ_FILE}" ] || rm -f "${REQ_FILE}"; }
trap cleanup EXIT

if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
  echo "==> Installing Miniconda into ${CONDA_ROOT} (no root needed)"
  TMP_INSTALLER="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "${TMP_INSTALLER}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "${TMP_INSTALLER}"
  else
    echo "Error: curl or wget is required to download Miniconda." >&2
    exit 1
  fi
  bash "${TMP_INSTALLER}" -b -p "${CONDA_ROOT}"
else
  echo "==> Reusing existing Miniconda at ${CONDA_ROOT}"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Creating conda env '${ENV_NAME}' (Python ${PYVER}, ffmpeg, git)"
  conda create -y -n "${ENV_NAME}" -c conda-forge "python=${PYVER}" pip ffmpeg git
else
  echo "==> Reusing existing conda env '${ENV_NAME}'"
fi

PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
export PYTHONNOUSERSITE=1
if ! "${PY}" -m pip --version >/dev/null 2>&1; then
  "${PY}" -m ensurepip --upgrade || conda install -y -n "${ENV_NAME}" -c conda-forge pip
fi

echo "==> Installing Python dependencies with CUDA JAX"
"${PY}" -m pip --disable-pip-version-check install -U pip "setuptools<82" wheel
"${PY}" -m pip --disable-pip-version-check install "numpy<2"
"${PY}" -m pip --disable-pip-version-check install -e "${REPO_ROOT}/third_party/gym-pybullet-drones"
REQ_FILE="$(mktemp /tmp/dreamerv3-nojax.XXXXXX.txt)"
grep -vE '^(jax|jaxlib|optax|nvidia-cuda-nvcc-cu12)' \
  "${REPO_ROOT}/third_party/dreamerv3/requirements.txt" > "${REQ_FILE}"
"${PY}" -m pip --disable-pip-version-check install -r "${REQ_FILE}"
"${PY}" -m pip --disable-pip-version-check install \
  "optax==0.2.4" "jax[cuda12]==0.4.33" "nvidia-cuda-nvcc-cu12>=12.8,<13"
"${PY}" -m pip --disable-pip-version-check install "tensorflow-cpu<2.16" tensorboard

echo "==> Running GPU sanity check"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"
"${PY}" - <<'PY'
import jax, jax.numpy as jnp
import optax, gym_pybullet_drones, gymnasium, embodied, elements
import pkg_resources
import tensorflow as tf
print("jax:", jax.__version__, "optax:", optax.__version__, "tf:", tf.__version__)
print("jax devices:", jax.devices())
probe = (jnp.ones((32, 32)) @ jnp.ones((32, 32))).sum()
print("jax compile probe:", float(probe.block_until_ready()))
import drone_nav
print("drone_nav import OK")
PY

echo "==> Setup complete."
echo "    Activate: source ${CONDA_ROOT}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
echo "    Train:    bash cloud/train.sh"
