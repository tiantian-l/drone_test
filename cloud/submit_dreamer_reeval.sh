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
time_limit="${EVAL_TIME_LIMIT:-02:00:00}"
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
  local out="$out_root/$scenario/$label"
  if [[ -e "$out" ]]; then echo "skip $label -> $scenario: $out exists"; return; fi
  local cmd=(sbatch --parsable --export=ALL --time="$time_limit" --job-name="reeval_${label}_${scenario}"
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
