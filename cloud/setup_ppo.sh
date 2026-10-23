#!/usr/bin/env bash
# Run from any directory. Creates an isolated Python 3.11 environment by default.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
ppo_python="${PPO_PYTHON:-python3.11}"
ppo_venv="${PPO_VENV:-${repo_root}/.venv/ppo}"
if [[ ! -x "${ppo_venv}/bin/python" ]]; then
  "${ppo_python}" -m venv "${ppo_venv}"
fi
"${ppo_venv}/bin/python" -m pip install -r "${repo_root}/requirements-ppo.txt"
"${ppo_venv}/bin/python" -m pip check
# Entrypoints use the vendored drone source directly; no editable install or JAX changes.
printf 'PPO Python: %s\n' "${ppo_venv}/bin/python"
