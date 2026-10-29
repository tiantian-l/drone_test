#!/usr/bin/env bash
# Re-evaluate the selected DreamerV3 checkpoints on the PID-reset test suite.
# Run with bash from the repository root on a login node (not with sbatch).
# Checkpoints come from cloud/dreamer_checkpoints.tsv; each checkpoint name is
# located inside its run directory (prefers best_checkpoints_freshenv_rng0_v2_128maps).
#
#   DRY_RUN=1 bash cloud/submit_dreamer_reeval.sh     # print the 21 submissions
#   bash cloud/submit_dreamer_reeval.sh
#
# Matrix (21 jobs): pi1..pi4 x {10_sparse,10_dense,20_sparse,20_dense};
# pi3 additionally on 20_sparse_dynamic; pi_dyn and pi_scratch on
# {20_sparse, 20_sparse_dynamic}. Outputs:
#   $OUT_ROOT/<scenario>/<label>/   (never overwritten; existing dirs are skipped)
set -euo pipefail
cd "$(dirname "$0")/.."
table="${TABLE:-cloud/dreamer_checkpoints.tsv}"
suite="${TEST_SUITE_DIR:-$PWD/cloud/test_maps_pidreset_v2}"
out_root="${OUT_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval_pidreset_v2}"
# Short requests schedule (backfill) much faster. Per-scenario defaults; override
# with EVAL_TIME_LIMIT (all jobs), TIME_10M / TIME_20M, and EVAL_CPUS.
time_10m="${TIME_10M:-00:30:00}"
time_20m="${TIME_20M:-01:00:00}"
cpus="${EVAL_CPUS:-16}"
[[ -f "$table" ]] || { echo "Missing $table" >&2; exit 1; }
resolve() {  # run_dir checkpoint_name|path -> checkpoint path
  local run_dir="$1" name="$2"
  if [[ -e "$name" ]]; then echo "$name"; return; fi
  local hit
  hit=$(find "$run_dir" -maxdepth 3 -name "$name" -path '*best_checkpoints_freshenv_rng0_v2_128maps*' -print -quit)
  [[ -n "$hit" ]] || hit=$(find "$run_dir" -maxdepth 3 -name "$name" -print -quit)
  [[ -n "$hit" ]] || { echo "Checkpoint $name not found under $run_dir" >&2; return 1; }
  echo "$hit"
}
static=(10_sparse 10_dense 20_sparse 20_dense)
declare -A run ckpt
while IFS=$'\t' read -r label scen run_dir checkpoint _; do
  [[ -z "$label" || "$label" == \#* ]] && continue
  run[$label]="$run_dir"; ckpt[$label]="$(resolve "$run_dir" "$checkpoint")"
  echo "$label: ${ckpt[$label]}"
done < "$table"
for label in pi1 pi2 pi3 pi4 pi_dyn pi_scratch; do
  [[ -n "${run[$label]:-}" ]] || { echo "Missing $label in $table" >&2; exit 1; }
done
for scenario in "${static[@]}" 20_sparse_dynamic; do
  [[ -f "$suite/$scenario.json" ]] || { echo "Missing manifest $suite/$scenario.json" >&2; exit 1; }
done
submit() {
  local label="$1" scenario="$2"
  local base="$out_root/$scenario/$label" out n=1
  # A finished result has a summary; skip it. A directory without one is a
  # failed/timed-out attempt: keep it and write the retry to <label>_retryN.
  if compgen -G "$base*/*__summary.json" > /dev/null; then echo "skip $label -> $scenario: finished"; return; fi
  out="$base"
  while [[ -e "$out" ]]; do out="${base}_retry${n}"; n=$((n + 1)); done
  local time_limit="${EVAL_TIME_LIMIT:-$([[ $scenario == 10_* ]] && echo "$time_10m" || echo "$time_20m")}"
  local cmd=(sbatch --parsable --export=ALL --time="$time_limit" --cpus-per-task="$cpus" --job-name="reeval_${label}_${scenario}"
             cloud/eval_final_slurm.sh "$scenario" "${run[$label]}" "${ckpt[$label]}" "$out")
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "POLICY_LABEL=${label}_pidreset TEST_SUITE_DIR=$suite ${cmd[*]}"
  else
    mkdir -p "$out_root/$scenario"
    local id; id=$(POLICY_LABEL="${label}_pidreset" TEST_SUITE_DIR="$suite" "${cmd[@]}")
    echo "$label -> $scenario: job $id"
  fi
}
for label in pi1 pi2 pi3 pi4; do for scenario in "${static[@]}"; do submit "$label" "$scenario"; done; done
submit pi3 20_sparse_dynamic
for label in pi_dyn pi_scratch; do for scenario in 20_sparse 20_sparse_dynamic; do submit "$label" "$scenario"; done; done
