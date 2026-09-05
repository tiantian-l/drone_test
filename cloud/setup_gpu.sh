#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu + CUDA GPU instance (RunPod / Vast.ai / GCP).
# Usage (on the remote machine, inside the repo root):
#   bash cloud/setup_gpu.sh
set -euo pipefail

# DMLab/MineRL need <=3.11; gym-pybullet-drones + DreamerV3 also happy on 3.11.
PYVER=3.11
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Installing Python ${PYVER} and system deps"
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python${PYVER}-dev python${PYVER}-venv ffmpeg git

echo "==> Creating virtualenv"
python${PYVER} -m venv "${REPO_ROOT}/.venv-gpu"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv-gpu/bin/activate"
export PYTHONNOUSERSITE=1
python -m pip install -U pip "setuptools<82" wheel

echo "==> Installing gym-pybullet-drones (drone env only)"
# numpy<2 is required by DreamerV3; install it first to pin the resolver.
python -m pip install "numpy<2"
python -m pip install -e "${REPO_ROOT}/third_party/gym-pybullet-drones"

echo "==> Installing DreamerV3 requirements with CUDA JAX"
# Keep JAX and Optax mutually compatible, and replace the requirements file's
# legacy <=12.2 CUDA compiler pin with CUDA 12.8 for Blackwell/SM 120 support.
grep -vE '^(jax|jaxlib|optax|nvidia-cuda-nvcc-cu12)' "${REPO_ROOT}/third_party/dreamerv3/requirements.txt" > /tmp/req-nojax.txt
python -m pip install -r /tmp/req-nojax.txt
python -m pip install "optax==0.2.4" "jax[cuda12]==0.4.33" "nvidia-cuda-nvcc-cu12>=12.8,<13"
# TensorBoardOutput imports TensorFlow; use the CPU build because JAX owns GPU.
python -m pip install "tensorflow-cpu<2.16" tensorboard

echo "==> Sanity check"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/dreamerv3:${PYTHONPATH:-}"
python - <<'PY'
import jax, jax.numpy as jnp
import optax, gym_pybullet_drones, gymnasium, embodied, elements
import pkg_resources  # noqa: required by gym-pybullet-drones at runtime
import tensorflow as tf  # noqa: needed by elements TensorBoardOutput
print("jax:", jax.__version__, "optax:", optax.__version__, "tf:", tf.__version__)
print("jax devices:", jax.devices())
probe = (jnp.ones((32, 32)) @ jnp.ones((32, 32))).sum()
print("jax compile probe:", float(probe.block_until_ready()))
import drone_nav  # noqa
print("drone_nav import OK")
PY

echo "==> Setup complete. Activate with: source .venv-gpu/bin/activate"
