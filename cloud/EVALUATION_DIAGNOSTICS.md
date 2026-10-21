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

## Offline analysis and plots

No GPU, checkpoint loading, or simulator is needed. The analysis script uses only
Python's standard library. Plotting requires NumPy and Matplotlib. Run from the
repository root with the cluster's existing Python environment:

```sh
RESULT=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval/10_dense/job_2096113
python cloud/analyze_diagnostics.py "$RESULT"
head -n 6 "$RESULT/analysis/failures.csv"
python cloud/plot_diagnostic_episode.py "$RESULT" --seed 1010000
```

Replace the example seed with one listed in failures.csv. Plotting accepts any
recorded seed, including successes. The default image path is
`RESULT/plots/seed_SEED.png`. Use `--output /path/to/new.png` for another location.
The analyzer also accepts `--output /path/to/new_analysis_directory`. Existing
outputs are rejected to protect review work. Input can instead be the specific
`*__diagnostics` directory, especially when a parent contains multiple captures.

Analysis requires status=complete and mode=full. It validates file counts, worker
quotas, unique seeds, geometry fingerprints, trajectory boundaries, step/time
alignment, terminal state and path length. If available, it checks the matching
original summary and per-map episode outcomes. Missing reference summaries are
explicitly reported rather than represented as a successful comparison.

Outputs: episodes.csv (all maps), failures.csv (success=0), summary.json (counts,
rates, provenance and validation status). Terminal event flags can overlap;
behavior_label starts as unreviewed. Collision IDs use the recorded safety-distance
threshold; negative-clearance IDs indicate overlap, not measured contact force.
Mean speed is over post-action samples. Minimum measured clearance is blank when
nothing enters the finite query radius, and must not be compared as if every
trajectory had fully observed clearances.

The six-panel image shows top-view obstacle footprints and flight path, altitude,
speed, velocity components, finite-radius clearance, and roll/pitch. Moving
obstacle paths and initial outlines are shown when present; this is not an
instantaneous collision visualization. Command at row t applies over (t-1,t].
Unknown clearance is marked at the query radius as a lower bound, with gaps in
the measured line. No causal failure labels are inferred automatically.

## Recorded 3D video replay

MP4 playback uses existing full diagnostic map/trajectory files, not a new policy
rollout. CPU only; requires NumPy, Matplotlib and the FFmpeg executable with libx264
(e.g. `conda install numpy matplotlib ffmpeg` in the analysis environment).
The CSV/summary exports alone are insufficient: retain the original diagnostics.

```sh
RESULT=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval/10_dense/job_2096113
python cloud/replay_diagnostic_episode.py "$RESULT" --seed 1010145
# A slower selected interval, saved separately:
python cloud/replay_diagnostic_episode.py "$RESULT" --seed 1010145 --start 4 --speed 0.5 --output "$RESULT/videos/seed_1010145_slow.mp4"
```

Default output: `RESULT/videos/seed_SEED.mp4` plus `.camera.json` containing actual
camera position, target, FOV, playback settings and source identity. Existing
files are never overwritten. Supported options: `--fps 20`, `--speed 1`,
`--start 0`, `--end SECONDS`, `--tail 1`, `--hold 1`, `--width 1280`,
`--elevation 52`, `--height-ratio 0.8`, `--azimuth DEGREES`.

Camera height initially uses 0.8 times the full horizontal scene extent, including
bounds, start/goal, obstacles and the recorded path. Target height is the midpoint
of permitted flight heights. Horizontal distance follows from that height
 difference and the selected depression angle. Default azimuth is perpendicular to the flight-region edge nearest the start.
The camera sits behind that edge midpoint, looking toward the flight-region
center, independent of randomized start/goal offsets. It stays at least 0.3 scene
lengths beyond the edge; height is increased if needed to preserve the requested
elevation. Explicit --azimuth overrides this automatic edge selection. Camera and FOV stay fixed throughout the clip;
if geometry extends behind the initial camera, it retreats along its view ray.
FOV is fitted to the full scene with margin; actual camera values are recorded.

Gray boxes are static; orange cylinders move using each recorded step. A schematic
enlarged drone cross follows recorded roll/pitch/yaw. Obstacles are translucent,
and the drone/trail are deliberate x-ray overlays to aid diagnosis: this is not a
photorealistic simulator render or a sensor-visibility reconstruction. Green circle
and star mark start and goal. The HUD shows time, altitude, speed, clearance, and
terminal flags at the recorded terminal state. Full captures include both static
and dynamic cases. Playback holds the preceding recorded state at each output
frame (no invented interpolation), always includes the selected end state, and
holds it for the requested duration. Arbitrary clips ending early do not label
that endpoint as a failure. Clearance outside query radius is a lower bound.

Run tests: `python -m unittest discover -s cloud -p 'test_diagnostic_replay.py'`.
Synthetic geometry/MP4 checks do not replace reviewing real recorded episodes.

## Batch all failure plots and videos

```sh
python cloud/render_failed_episodes.py "$RESULT"
```

Reads unsuccessful episode records directly (no failures.csv required), sorts seeds,
and sequentially invokes the existing plot and 3D replay scripts using the current
Python interpreter. Default outputs: `RESULT/failure_media/plots`, `videos`, and
`batch.jsonl`. Each operation has a `.log` file. Successful existing artifacts are
skipped (MP4 also requires its camera metadata). Individual errors do not stop later
cases; the final process exits nonzero if any operation failed. Partial files are
never automatically deleted. Use a fresh `--output` directory to regenerate with
new camera/render settings; skips do not compare those settings. Optional video
settings: `--fps 20 --speed 0.5 --width 1280`. No GPU is required; run long CPU
rendering batches on a compute allocation according to cluster policy.

### Multi-view trial version

Replay now defaults to `--layout multi`: fixed-camera 3D on the left,
synchronized top view at upper right, and a fixed local vertical projection at
lower right. Static and moving obstacles share a height color scale. Obstacles
whose recorded terminal clearance crosses the collision threshold are red
throughout the retrospective replay. This highlight is hindsight, not detection.
The local projection follows the terminal drone-to-obstacle horizontal bearing;
other obstacles are omitted when a terminal obstacle is identified. Overlap in
this projection alone is not contact. No nearest-surface contact coordinates are
inferred. Terminal X marks drone position. For boundary failures without a
terminal obstacle the side view defaults to the x-z plane; top view retains the
full trajectory extent, including boundary exits.

Failure episodes play the last two seconds at half speed by default. Configure
`--slow-last 2 --slow-factor 0.5`; use `--slow-last 0` to disable. Use `--hold 3`
for a longer terminal pause, and `--layout single` for the original layout.
Example (new filename avoids overwriting earlier videos):

```sh
python cloud/replay_diagnostic_episode.py "$RESULT" --seed 1010049 --layout multi --hold 3 --output "$RESULT/videos/seed_1010049_multi.mp4"
```

Sync both replay_diagnostic_episode.py and replay_views.py. Batch rendering uses
the new default layout, but existing videos are skipped: choose a new batch output
directory to regenerate. This trial still uses projected polygon rendering;
it does not yet replace the renderer with a depth-buffered simulator engine.

### Terminal-region close-up update

The lower-right panel is now a true 3D orthographic close-up, replacing the side
projection. It uses a fixed 4 m x 4 m x 3 m window around the terminal drone
position, at 28 degrees elevation and a side-oblique bearing relative to the
terminal obstacle. It activates during the final two recorded seconds. The first
two panels remain unchanged. Target obstacles are red with their full height
labels; other nearby obstacles are faint. Objects may be cropped by the local
window, so displayed cut faces are not physical obstacle tops. The blue marker
and trail are recorded drone-center positions; the terminal cross is not a contact
point. Relevant obstacle clearance is taken directly from the recorded query, or
marked not measured. For boundary failures, red dashed lines indicate horizontal
flight boundaries at current drone altitude. Use a fresh output filename to test
this update, and sync both replay_diagnostic_episode.py and replay_views.py.
