#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu + CUDA GPU instance (RunPod / Vast.ai / GCP).
# Usage (on the remote machine, inside the repo root):
#   bash cloud/setup_gpu.sh
set -euo pipefail

# DMLab/MineRL need <=3.11; gym-pybullet-drones + DreamerV3 also happy on 3.11.
PYVER=3.11

echo "==> Installing Python ${PYVER} and system deps"
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python${PYVER}-dev python${PYVER}-venv ffmpeg git

echo "==> Creating virtualenv"
python${PYVER} -m venv .venv-gpu
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
pip install -U pip setuptools wheel

echo "==> Installing gym-pybullet-drones (drone env only)"
# numpy<2 is required by DreamerV3; install it first to pin the resolver.
pip install "numpy<2"
pip install -e third_party/gym-pybullet-drones

echo "==> Installing DreamerV3 requirements with CUDA JAX"
# Replace the CPU JAX line with the CUDA wheel for the GPU box.
grep -v '^jax' third_party/dreamerv3/requirements.txt > /tmp/req-nojax.txt
pip install -r /tmp/req-nojax.txt
pip install "jax[cuda12]==0.4.33"

echo "==> Sanity check"
python - <<'PY'
import jax, gym_pybullet_drones, gymnasium, embodied, elements
print("jax devices:", jax.devices())
import drone_nav  # noqa
print("drone_nav import OK")
PY

echo "==> Setup complete. Activate with: source .venv-gpu/bin/activate"
