# Evaluation diagnostics

Opt-in shared recording for checkpoint validation, frozen test manifests, and
training-time evaluation. Defaults remain off. No fields are added to policy
observations, and reward, termination, seeds, and evaluation quotas are unchanged.
The recorder is UAV-specific. It runs inside each environment worker, records
both successes and failures, and stops recording after that worker's quota.

## Independent evaluation

Add `--diagnostics full` (or `summary`, `off`) to either existing command:

```sh
python cloud/eval_checkpoint.py --config RUN/config.yaml --checkpoint CHECKPOINT --output NEW_OUTPUT --eval-maps 128 --diagnostics full
python cloud/eval_final.py --config RUN/config.yaml --checkpoint CHECKPOINT --manifest cloud/test_maps_v1/20_sparse.json --output NEW_OUTPUT --diagnostics full
```

Validation keeps the saved validation map schedule. Test keeps the frozen manifest
and 256-map coverage. Existing output files remain unchanged. Diagnostics are in
an additional directory. New run directories must not exist; accidental overwrites
are rejected. The source config and evaluation config identify training and target
environments; test metadata additionally contains the policy label and scenario.

## During training

Use these config overrides with the existing training entry point:

```text
--run.eval_diagnostics full
--run.eval_diagnostics_every 5
--run.eval_diagnostics_snapshot True
```

This records full trajectories on cycles 0, 5, 10, ... and summaries on all other
cycles. Set `every=1` for every cycle. `summary` always records summaries; `off`
disables diagnostics entirely. Each cycle gets its own step/cycle directory under
`LOGDIR/diagnostics`. Enabled cycles save an agent-only snapshot by default inside
the evaluation context, including low-performing cycles that Top-K would discard.
Snapshots are not automatically pruned and can consume substantial disk space.
Set snapshot False to omit them; run/step identification alone then does not
promise that those weights will remain replayable. Full training-resume state
(replay and optimizer history) is not promised by these agent-only snapshots.

## Files and alignment

- `run.json`: schema, source, split, checkpoint/cycle identity, mode, protocol,
  quota and status. Only `status=complete` is a completed validated capture.
- `worker_*_episode_*_seed_*.map.json`: frozen initial geometry, goal, bounds,
  body ID mapping and sensor/control constants.
- `*.episode.json`: outcome logs, executed step count, path length, terminal state,
  horizontal/vertical boundary and attitude flags, nearby obstacle IDs/clearance.
- `*.trajectory.jsonl` (full only): initial sample plus one sample per executed
  step, original model observations, drone state, command, reward, boundary margins,
  dynamic geometry and nearby obstacle distances. Initial action is null. At row
  t>0 the action was executed from row t-1 and produced the state at row t.
  Submitted and post-wrapper executed actions are recorded separately where the
  simulator's action buffer is available. Target velocity follows ActionType.VEL.

State order follows `_getDroneStateVector`: position 0:3, quaternion 3:7,
roll/pitch/yaw 7:10, linear velocity 10:13, angular velocity 13:16, RPM 16:20.
Nearby clearance queries have a recorded finite radius (at least 1 m). An empty
list means no surface within that radius, NOT infinite known clearance. Negative
signed clearance indicates geometric overlap; actual contact-force events are
not recorded. Existing collision flags retain the 0.15 m safety-distance meaning.
Events may overlap. Samples occur at control rate, not physics substep rate.

Behavior labels such as late avoidance, oscillation or poor route choice are not
assigned automatically: these require a separate, versioned offline classifier
and review. Raw observations are retained to support that later analysis. Sensor
ray hit IDs, model latent states, causal interventions and exposure-normalized
classification tables are not part of this recorder.

## Validation

Run the recorder and protocol tests using a Python environment with NumPy:

```sh
python -m unittest discover -s cloud -p 'test_evaluation*.py'
```

Tests cover unchanged mock outputs, action/state alignment, quota suppression,
initial geometry freezing, overlapping event flags and summary/full modes, plus
the existing shared evaluation lifecycle tests. These do not establish GPU or
PyBullet replay equivalence. Before production collection, run the same checkpoint
and full map schedule with recording off and full, and compare per-map results.
Also cross two training evaluation cycles and verify separate complete directories
and reloadable snapshots. Recording adds CPU queries and filesystem overhead.

## Slurm submission

From the repository root on the cluster:

```sh
EVAL_DIAGNOSTICS=full sbatch cloud/eval_final_slurm.sh 20_dense /absolute/training/run /absolute/checkpoint
```

The optional fourth positional argument is a new output directory. Without it,
results go to the existing project `final_eval/SCENARIO/job_JOBID` directory.
`EVAL_DIAGNOSTICS` accepts off (default), summary, or full. The launcher defaults
to `cloud/test_maps_diagnostics`, containing the same seeds and scenario definitions
as `test_maps_v1`, with updated source fingerprints for the three diagnostic run
settings only. Original manifests remain untouched. `TEST_SUITE_DIR` can override
the directory, but source checks remain enforced. Sync the launcher, new manifests,
recorder, and modified Dreamer submodule files to the cluster together.
