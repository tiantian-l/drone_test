#!/usr/bin/env bash
# Quick Mac (CPU) smoke training with rendered third-person video.
# Lets you SEE the drone navigating in TensorBoard while it trains.
#
#   bash scripts/mac_test_train.sh            # uses the env that has the deps
#   PYTHON=/path/to/python bash scripts/mac_test_train.sh
#
# Then open TensorBoard (printed at the end) and look at:
#   IMAGES/VIDEO -> policy_log/image   (third-person flight)
#   SCALARS      -> episode/score, epstats/log/is_success/max, epstats/log/distance/avg
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Default to the conda env that already has the deps; override with PYTHON=...
DEFAULT_PY="${HOME}/miniconda3/envs/drone/bin/python"
PYTHON="${PYTHON:-${DEFAULT_PY}}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="python"
fi

# drone_nav lives at the repo root; expose it so `import drone_nav` resolves
# when main.py runs from inside third_party/dreamerv3.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Headless PyBullet rendering + avoid a known macOS OpenMP duplicate-lib crash.
export KMP_DUPLICATE_LIB_OK=True

TS="$(date +%Y%m%dT%H%M%S)"
LOGDIR="${LOGDIR:-${HOME}/logdir/drone_nav_mactest/${TS}}"

echo "==> Logging to: ${LOGDIR}"
echo "==> After it starts, in another terminal run:"
echo "    tensorboard --logdir ${HOME}/logdir/drone_nav_mactest --port 6006"
echo "    then open http://localhost:6006  (IMAGES tab -> policy_log/image)"
echo

cd "${REPO_ROOT}/third_party/dreamerv3"
exec "${PYTHON}" dreamerv3/main.py \
  --configs drone_nav_mactest \
  --logdir "${LOGDIR}" \
  "$@"
