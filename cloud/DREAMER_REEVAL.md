# DreamerV3 re-evaluation after the PID-reset fix

The selected checkpoints are **not retrained and not re-selected**; only the
test environment changes (per-episode PID reset; `nav_aviary.py` byte-identical to
`feature/apf` and `feature/A_star`). Test maps: `cloud/test_maps_pidreset_v2/` (same seeds as
`test_maps_v1`, new source fingerprint), the same suite used for Map-A*.

Checkpoints: `cloud/dreamer_checkpoints.tsv` (label, training scenario, run
directory, checkpoint name). Training details: RTX 4090, batch size 16.

## 1. Submit 21 evaluation jobs

```bash
DRY_RUN=1 bash cloud/submit_dreamer_reeval.sh   # inspect
bash cloud/submit_dreamer_reeval.sh
```

| Policies | Test conditions | Jobs |
|---|---|---|
| π1–π4 | E1–E4 | 16 |
| π3 | E_dyn | 1 |
| π_dyn, π_scratch | E3, E_dyn | 4 |

Uses `cloud/eval_final_slurm.sh` unchanged (16 workers, bf16, A100,
deterministic policy mean, protocol `freshenv_rng0_v2`), with
`TEST_SUITE_DIR=cloud/test_maps_pidreset_v2`. Results:
`/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval_pidreset_v2/<scenario>/<label>/`.
Existing output directories are skipped, so the script can be re-run to submit
only missing jobs.

## 2. Compare with the original results

```bash
python cloud/compare_reeval.py --output reeval_report.md
```

Old vs new success on the same maps, maps gained/lost, exact McNemar p,
failures within the first second (where stale PID state acted), geometry-hash
check, and the updated Tables 4.2–4.6. Decision rule agreed beforehand: if the
differences are within noise (no significant McNemar result, few pp), keep the
trained policies and report the re-evaluated numbers; otherwise consider
retraining.
