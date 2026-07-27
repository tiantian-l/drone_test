#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
variants_csv="${VARIANTS:-a,b,c,d}"
seeds_csv="${SEEDS:-0,1,2}"
log_root="${LOG_ROOT:-${repo_root}/logs/ablation}"
task_python="${PYTHON:-python3}"
jax_platform="${JAX_PLATFORM:-cuda}"
dry_run="${DRY_RUN:-0}"
main_py="${repo_root}/third_party/dreamerv3/dreamerv3/main.py"

IFS=',' read -r -a variants <<< "${variants_csv}"
IFS=',' read -r -a seeds <<< "${seeds_csv}"

for variant in "${variants[@]}"; do
  case "${variant}" in
    a|A) variant=a ;;
    b|B) variant=b ;;
    c|C) variant=c ;;
    d|D) variant=d ;;
    *) echo "Unknown variant '${variant}'; use a,b,c,d" >&2; exit 2 ;;
  esac
  for seed in "${seeds[@]}"; do
    case "${variant}" in
      a) variant_dir=A ;;
      b) variant_dir=B ;;
      c) variant_dir=C ;;
      d) variant_dir=D ;;
    esac
    logdir="${log_root}/${variant_dir}/seed_${seed}"
    cmd=(
      "${task_python}" "${main_py}"
      --configs drone_nav "drone_ablation_${variant}"
      --seed "${seed}"
      --logdir "${logdir}"
      --jax.platform "${jax_platform}"
      --logger.outputs jsonl,scope,tensorboard
      "$@"
    )
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "${dry_run}" != "1" ]]; then
      (
        cd "${repo_root}/third_party/dreamerv3"
        PYTHONPATH="${repo_root}:${repo_root}/third_party/dreamerv3:${PYTHONPATH:-}" \
          "${cmd[@]}"
      )
    fi
  done
done
