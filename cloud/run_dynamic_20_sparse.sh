#!/usr/bin/env bash
# New dynamic run, static-agent initialization, or explicit dynamic resume.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-drone}"
PY="${TRAIN_PYTHON:-${CONDA_ROOT}/envs/${ENV_NAME}/bin/python}"
SEED="${SEED:-0}"
STEPS="${STEPS:-3000000}"
LOGDIR="${LOGDIR:-${LOG_ROOT:-$HOME/logdir/drone_dynamic_20_sparse}/seed_${SEED}}"
MODE="${MODE:-scratch}"
if [[ ! -x "$PY" ]]; then
  echo "Training Python missing: $PY. Run cloud/setup.sh first." >&2; exit 2
fi
if [[ ! "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS must be a positive integer (total dynamic environment steps)." >&2; exit 2
fi
LOGDIR="$("$PY" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$LOGDIR")"
export PYTHONNOUSERSITE=1
export PATH="$(dirname "$PY"):$PATH"
initialization=()
case "$MODE" in
  scratch|finetune)
    if [[ -d "$LOGDIR" ]] && [[ -n "$(ls -A "$LOGDIR")" ]]; then
      echo "New runs require an empty LOGDIR; use MODE=resume for an existing dynamic run." >&2; exit 2
    fi
    if [[ "$MODE" == finetune ]]; then
      if [[ -z "${INIT_CHECKPOINT:-}" || ! -e "$INIT_CHECKPOINT" ]]; then
        echo "MODE=finetune requires INIT_CHECKPOINT pointing to a compatible agent checkpoint." >&2; exit 2
      fi
      INIT_CHECKPOINT="$("$PY" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$INIT_CHECKPOINT")"
      initialization=(--run.from_checkpoint "$INIT_CHECKPOINT")
    fi
    ;;
  resume)
    if (( $# )); then
      echo "Resume preserves saved settings; extra CLI overrides are not accepted." >&2; exit 2
    fi
    if [[ ! -d "$LOGDIR/ckpt" || ! -f "$LOGDIR/config.yaml" ]]; then
      echo "MODE=resume requires an existing dynamic LOGDIR with config.yaml and ckpt/." >&2; exit 2
    fi
    # Use the saved configuration verbatim, changing only the target step budget.
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/dreamerv3:${PYTHONPATH:-}"
    "$PY" "$REPO_ROOT/cloud/resume_dynamic.py" "$LOGDIR" "$STEPS" "${DRY_RUN:-0}"
    exit
    ;;
  *) echo "MODE must be scratch, finetune, or resume." >&2; exit 2 ;;
esac
export PYTHONNOUSERSITE=1
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/dreamerv3:${PYTHONPATH:-}"
cmd=("$PY" "$REPO_ROOT/third_party/dreamerv3/dreamerv3/main.py"
  --configs drone_nav drone_perc_cnn_hi drone_static_20_sparse drone_dynamic_20_sparse drone_reward_v2
  --seed "$SEED" --logdir "$LOGDIR" --run.steps "$STEPS"
  --jax.platform "${JAX_PLATFORM:-cuda}" --jax.compute_dtype "${JAX_COMPUTE_DTYPE:-bfloat16}"
  --run.debug False --run.report_every 1800 --run.eval_every_steps 100000
  --logger.outputs jsonl,scope,tensorboard
  --env.drone.log_image "${LOG_IMAGE:-False}" --env.drone.video_every 100
  "$@")
if [[ "$MODE" == finetune ]]; then
  cmd+=("${initialization[@]}")
fi
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  cd "$REPO_ROOT/third_party/dreamerv3"
  exec "${cmd[@]}"
fi
