# Fresh-environment evaluation, version 2

Protocol: `freshenv_rng0_v2`. Training and independent evaluation use
`embodied/run/evaluation.py`. Every cycle creates fresh evaluation environments,
initializes policy carry under the current-weight snapshot, starts the policy RNG
counter at zero, and closes evaluation workers even on callback failure. Training
environments and the training action counter are not reset.

The four static presets now use 16 evaluation workers x 8 maps = 128 episodes.
Seed bases and stride (16) are unchanged; slots 0..3 preserve the old map set,
and slots 4..7 add 64 maps. Geometry, rewards, training batch size and training
environment count are unchanged. CLI overrides still take precedence.

Training writes per-cycle ordered episode records and summaries into
`evaluation_freshenv_rng0_v2_128maps`. Records include worker, slot, seed,
episode index, policy RNG counter at episode start, episode length and terminal
scalar logs. Summaries include outcome counts/rates, original/added map group
rates, unique map count, seed-list SHA256, and elapsed time. The seed-list hash
is NOT a geometry or weight hash. TensorBoard receives numeric `eval_audit/*`
metrics. Historical `eval_maps.jsonl` remains available, sorted by seed.

Best checkpoints use `best_checkpoints_freshenv_rng0_v2_128maps`, separate from
legacy scores. The regular full checkpoint format and replay restore are unchanged.
Resume a copied run for validation; this change does not retroactively correct old
checkpoint scores. Saved config files are not automatically upgraded by the
independent evaluator: pass `--eval-maps 128`, or submit the evaluation Slurm script
with `EVAL_MAPS=128`. Without an override it preserves the saved map count.

Costs and limitations:
- Evaluation now has twice as many episodes plus worker startup overhead.
- Fresh environments reset video scheduling, so worker 0 can record its first
  episode every cycle when image logging is enabled. Disable LOG_IMAGE for the
  initial validation if that overhead is unwanted.
- Original-map group scores are useful diagnostics, not guaranteed bitwise
  equivalents to a separate 64-map evaluation: additional active worker episodes
  can alter batch context. Compare identical protocol and episode count.
- Fixed map coverage does not establish representativeness or statistical power.
- Geometry inventories, weight fingerprints and detailed crash subcategories are
  not implemented in this patch; no observation-space fields were added.
- CPU mock tests do not prove real JAX/PyBullet or GPU reproducibility.

Acceptance: on a copied run cross two evaluation cycles, independently evaluate
both saved checkpoints twice using the saved config, and compare per-map outcomes
and ordered episode metadata with the corresponding training audit files. Check
128 unique seeds, slots 0..7 per worker, and original/added group counts.
