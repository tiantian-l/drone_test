# CNN-PPO baseline

The default is a **10_sparse feasibility run**, not a tuned or completed result:
500,000 requested environment transitions, 8 workers, 128 fixed validation maps.
The same implementation supports all five scenes. A 10_sparse-trained policy
must be compared against a 10_sparse-trained Dreamer policy in cross-scene tests.
Use a matched budget and multiple independent training seeds for final claims.

## Installation and checks

Run commands from the repository root on a training machine. The installer creates
an isolated environment at `.venv/ppo`; it does not modify the Dreamer environment.
Python 3.11 must already be installed (`PPO_PYTHON` can name another interpreter).
GPU PyTorch availability depends on the host's wheel/driver combination; inspect
`torch.cuda.is_available()` on the allocated GPU node before requesting `--device cuda`.

```bash
bash cloud/setup_ppo.sh
.venv/ppo/bin/python -m unittest discover -s cloud -p 'test_ppo*.py' -v
```

The integration tests use actual PyBullet: an optimizer update, checkpoint
round-trip, timeout bootstrapping inputs, seed equivalence, and spawned evaluation
workers. They **skip** when dependencies are missing; skips do not establish that
training works. Protocol/config tests need only PyYAML.

## First training run

```bash
bash cloud/run_ppo.sh --scenario 10_sparse --seed 0 \
  --steps 500000 --output /absolute/path/ppo/10_sparse/seed_0
```

Use `--steps 8192` for a short pipeline check. It still evaluates the full 128-map
validation set at the end; this is an interface check, not a convergence claim.
`--serial` uses in-process environments for debugging. Production defaults use
spawned subprocesses and do not inherit the parent's simulator state.

For Slurm (supply your site's account and partition):

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION cloud/train_ppo_slurm.sh \
  --scenario 10_sparse --seed 0 --steps 500000 \
  --output /absolute/path/ppo/10_sparse/seed_0
```

`PPO_PYTHON=/absolute/path/to/python bash cloud/run_ppo.sh ...` selects an existing
installation. The supplied Slurm script requests 24 CPUs and one GPU.

## Environment and policy

Without `--source-config`, the code merges the existing Dreamer presets in order:
defaults, drone_nav, drone_perc_cnn_hi, selected static scene, optional dynamic
scene, drone_reward_v2. Nested reward overrides are merged rather than replacing
the whole reward dictionary. The resolved environment is saved in `config.json`.

For exact comparison to an existing run, add:

```text
--source-config /absolute/path/dreamer_run/config.yaml
```

This accepts a resolved Dreamer `drone_nav` config, checks the labeled scenario,
and retains its perception/reward settings. Inspect resolved settings with
`--dry-run` before training. Changing `--scenario 20_sparse` selects the larger
scene; the default PPO budget remains 500,000 until explicitly overridden.

The default observations are state `(9,)` and float LiDAR `(8,72)` in [0,1]. A
state MLP (128,128) receives fixed `sign(x)*log(1+abs(x))`; a circular azimuth CNN
uses elevation layers as channels and produces 256 features. Actor and critic
each have (256,256) hidden layers. This is a feed-forward baseline, not PPO-LSTM.
No online observation statistics or reward normalization are used. There is no
image `/255` normalization. Actual parameter count is saved in `model.json`.

The original action shape `(1,4)`, direction normalization, absolute speed
component, 2 m/s cap, DSL-PID, reward and termination remain in NavigationAviary.
SB3's Gaussian actions are clipped to the existing Box bounds. We do not change
this to a different three-dimensional velocity parameterization.

Training workers use the same `SeedSequence([seed, worker])` initialization as
Dreamer, only on the first reset. Later episodes continue their RNG streams.
Different policy episode lengths still produce different sampled trajectories.
Gymnasium `terminated` versus `truncated` is preserved by SB3's vector wrapper,
including terminal observations used to bootstrap time limits.

## Validation, tuning, and checkpoints

Edit a copy of `configs/ppo.yaml` and pass `--config /path/to/candidate.yaml` for
hyperparameter experiments. The defaults (learning rate 3e-4, rollout 512 per
worker, batch 256, 10 epochs, gamma .99, entropy 0, KL guard .03) are initial
candidates. Select hyperparameters on validation only; never tune on final tests.
Ordinary `--config` overrides are not applied during resume, which uses the saved
algorithm config; supported CLI runtime overrides still apply.

Validation uses fresh environments, normally 16 workers × 8 maps, deterministic
policy inference, and the scene's existing validation seed base. The callback
runs after completed PPO updates at the first rollout boundary crossing each
50,000-step interval, and after the final update. Training environments are not
reset for validation. Each worker contributes exactly its assigned map quota;
surplus episodes on faster workers are ignored. The earliest checkpoint wins
ties in validation success rate. Configuring smaller validation quotas is for
debugging only, not a comparable final experiment.

Outputs:

- `config.json`, `provenance.json`, `model.json`: task, algorithm, source hashes,
  software versions, actual device and model size.
- `logs/`: SB3 CSV and TensorBoard, including validation success/failure rates.
- `step_*.zip`: each evaluated model, including SB3 optimizer state.
- `best_model.zip`, `best_model.json`: validation-selected model and selection record.
- `last_model.zip`, `final_model.zip`: latest evaluated and final models.
- `validation/step_*/`: per-map records, geometry, summaries and confidence intervals.
- `completed.json`: requested and actual environment steps. Full rollouts may
  exceed the requested budget; report actual steps when comparing learning curves.

The 95% Wilson intervals describe map-level uncertainty for one model, not
variation across training seeds. Success/collision/crash flags retain the
environment's semantics and may overlap. Arrival time is among successes only.

Continue in a **new** directory, using an additional-step budget:

```bash
bash cloud/run_ppo.sh --resume /absolute/path/old_run/last_model.zip \
  --steps 500000 --output /absolute/path/new_run
```

The saved run's config and optimizer state are restored. Fresh training workers
and RNGs are initialized: this is continuation, not bitwise restoration of the
old rollout/physics state. The resumed policy is evaluated before collecting new
data and is eligible for the new run's best model. The previous run is unchanged.

## Cross-scene final evaluation

Evaluate a frozen validation-selected model separately on each scene:

```bash
.venv/ppo/bin/python cloud/eval_ppo.py \
  --checkpoint /absolute/path/ppo/10_sparse/seed_0/best_model.zip \
  --manifest cloud/test_maps_v1/20_dense.json \
  --output /absolute/path/ppo_test/train_10_sparse_test_20_dense
```

Repeat with `10_sparse`, `10_dense`, `20_sparse`, `20_dense`, and
`20_sparse_dynamic`. Each evaluates exactly 16 workers × 16 maps. Test geometry,
task physics, and dynamics follow the manifest, while training perception stays
fixed. The evaluator reuses the existing coverage validator and statistics,
rejects validation/test overlap, and exports a geometry hash with the same JSON
encoding as `eval_final.py`. Compare this hash with Dreamer exports.
PPO's execution protocol is named `ppo_deterministic_fixed_maps_v1`; it does not
claim Dreamer's policy-RNG audit fields. Final-test device defaults to CPU.

**Known pre-existing repository condition found during implementation:** the five
`test_maps_v1` manifests currently record a different hash of Dreamer's
`configs.yaml` from the working source. Neither the manifests nor the frozen
sources were changed by this implementation. Final evaluation intentionally
rejects that mismatch. Use the corresponding frozen code/config version, or
review and version a new shared suite and evaluate both methods against it.
Do not regenerate manifests merely to bypass the check or mix old/new scores.

## Verification status

Configuration and protocol tests can run on the current local environment.
Actual training requires the separate dependencies; local dependency installation
was not authorized during implementation. No PPO convergence, GPU throughput, or
final navigation performance has been measured here.
