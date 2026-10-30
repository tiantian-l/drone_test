#!/usr/bin/env bash
# Re-run the thesis DreamerV3 test evaluations with outcome diagnostics, so that
# crashes can be split into flight-region exits and attitude-limit violations.
# Same environment (no per-episode PID reset), manifests (test_maps_diagnostics),
# checkpoints and protocol as the thesis tables; diagnostics do not change
# observations, rewards or termination.
#
#   bash cloud/submit_dreamer_diag.sh            # 5 jobs: main comparison (pi_i in E_i, pi_scratch in E_dyn)
#   ALL=1 bash cloud/submit_dreamer_diag.sh      # 21 jobs: all RQ1-RQ3 pairs
#   DRY_RUN=1 ...                                # print only
# Output: $OUT_ROOT/<scenario>/<label>/  (finished pairs are skipped)
set -euo pipefail
cd "$(dirname "$0")/.."
L=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir
suite="${TEST_SUITE_DIR:-$PWD/cloud/test_maps_diagnostics}"
out_root="${OUT_ROOT:-$L/final_eval_diag}"
cpus="${EVAL_CPUS:-16}"
declare -A run name
run[pi1]=$L/drone_static_factorial_a100_bs32/job_1941496/10_sparse/seed_0; name[pi1]=step_0002300000_success_0.9766.ckpt
run[pi2]=$L/bs32/job_1983614/10_dense/seed_0;                             name[pi2]=step_0001700000_success_0.9375.ckpt
run[pi3]=$L/4090/20_sparse/seed_0;                                        name[pi3]=step_0002200000_success_0.9531.ckpt
run[pi4]=$L/4090/20_dense/seed_0;                                         name[pi4]=step_0004800000_success_0.9297.ckpt
run[pi_dyn]=$L/drone_nav/20260921T143217;                                 name[pi_dyn]=step_0002900000_success_0.8984.ckpt
run[pi_scratch]=$L/drone_nav/20261008T204751;                             name[pi_scratch]=step_0005200000_success_0.9297.ckpt
resolve() {
  local hit
  hit=$(find "${run[$1]}" -maxdepth 3 -name "${name[$1]}" -path '*best_checkpoints_freshenv_rng0_v2_128maps*' -print -quit)
  [[ -n "$hit" ]] || hit=$(find "${run[$1]}" -maxdepth 3 -name "${name[$1]}" -print -quit)
  [[ -n "$hit" ]] || { echo "Checkpoint ${name[$1]} not found under ${run[$1]}" >&2; exit 1; }
  echo "$hit"
}
declare -A done_pairs
submit() {
  local label="$1" scenario="$2" base out n=1
  [[ -n "${done_pairs[$label/$scenario]:-}" ]] && return
  done_pairs[$label/$scenario]=1
  base="$out_root/$scenario/$label"
  if compgen -G "$base*/*__summary.json" > /dev/null; then echo "skip $label -> $scenario: finished"; return; fi
  out="$base"; while [[ -e "$out" ]]; do out="${base}_retry${n}"; n=$((n + 1)); done
  local t; [[ $scenario == 10_* ]] && t="${TIME_10M:-00:30:00}" || t="${TIME_20M:-01:00:00}"
  local cmd=(sbatch --parsable --export=ALL --time="$t" --cpus-per-task="$cpus" --job-name="diag_${label}_${scenario}"
             cloud/eval_final_slurm.sh "$scenario" "${run[$label]}" "$(resolve "$label")" "$out")
  if [[ -n "${DRY_RUN:-}" ]]; then echo "${cmd[*]}"; return; fi
  mkdir -p "$out_root/$scenario"
  echo "$label -> $scenario: job $(EVAL_DIAGNOSTICS=summary POLICY_LABEL="${label}_diag" TEST_SUITE_DIR="$suite" "${cmd[@]}")"
}
submit pi1 10_sparse; submit pi2 10_dense; submit pi3 20_sparse; submit pi4 20_dense; submit pi_scratch 20_sparse_dynamic
if [[ -n "${ALL:-}" ]]; then
  for p in pi1 pi2 pi3 pi4; do for s in 10_sparse 10_dense 20_sparse 20_dense; do submit "$p" "$s"; done; done
  submit pi3 20_sparse_dynamic
  for p in pi_dyn pi_scratch; do for s in 20_sparse 20_sparse_dynamic; do submit "$p" "$s"; done; done
fi
