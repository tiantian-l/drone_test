#!/usr/bin/env bash
# Run with bash from the repository root, not with sbatch.
# Resubmit the five 20_sparse-policy and four 20_dense-policy evaluations.
set -euo pipefail
cd "$(dirname "$0")/.."
root=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/4090
sparse_run="$root/20_sparse/seed_0"
dense_run="$root/20_dense/seed_0"
sparse_ckpt="$sparse_run/best_checkpoints_freshenv_rng0_v2_128maps/step_0002200000_success_0.9531.ckpt"
dense_ckpt="$dense_run/best_checkpoints_freshenv_rng0_v2_128maps/step_0004800000_success_0.9297.ckpt"
# One hour remains a provisional budget until successful-job Elapsed is known.
time_limit="${EVAL_TIME_LIMIT:-01:00:00}"
if [[ ! "$time_limit" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo 'EVAL_TIME_LIMIT must use HH:MM:SS.' >&2
  exit 2
fi
for path in "$sparse_run/config.yaml" "$dense_run/config.yaml" "$sparse_ckpt" "$dense_ckpt"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing input: $path; no jobs submitted." >&2
    exit 1
  fi
done
for scenario in 10_sparse 10_dense 20_sparse 20_dense 20_sparse_dynamic; do
  if [[ ! -f "${TEST_SUITE_DIR:-cloud/test_maps_v1}/$scenario.json" ]]; then
    echo "Missing test manifest for $scenario; no jobs submitted." >&2
    exit 1
  fi
done
submit_eval() {
  local policy_label="$1" scenario="$2" run_dir="$3" checkpoint="$4"
  local job_id
  job_id=$(POLICY_LABEL="$policy_label" sbatch --parsable --export=ALL \
    --time="$time_limit" --job-name="${policy_label}_to_${scenario}" \
    cloud/eval_final_slurm.sh "$scenario" "$run_dir" "$checkpoint")
  printf '%s -> %s: job %s (limit %s)\n' "$policy_label" "$scenario" "$job_id" "$time_limit"
}
for scenario in 10_sparse 10_dense 20_sparse 20_dense 20_sparse_dynamic; do
  submit_eval 4090_20_sparse_seed_0 "$scenario" "$sparse_run" "$sparse_ckpt"
done
for scenario in 10_sparse 10_dense 20_sparse 20_dense; do
  submit_eval 4090_20_dense_seed_0 "$scenario" "$dense_run" "$dense_ckpt"
done
