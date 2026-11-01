# Map-A* reference baseline

A map-based classical baseline: online LiDAR occupancy grid + grid shortest-path
replanning + pure-pursuit tracking + a local LiDAR safety layer. It provides
explicit memory and global planning (the reactive APF baselines on branch
`feature/apf` are no longer used in the thesis). This branch is self-contained: it does not import APF or PPO
code.

## Information available to the planner

- The same observation as the DreamerV3 policies (goal displacement, velocity,
  attitude, 8×72 LiDAR).
- **Declared task information** (option A): the goal position in the world
  frame, given once per episode via `set_task`, and the flight region from the
  environment settings. This lets the planner place the flight-region boundary
  in its map. Report this in the thesis.
- No simulator obstacle geometry and no obstacle velocities.

### Variant without flight region (`--task-info none`, `TASK_INFO=none`)

Observation only, like DreamerV3: the map is built in a goal-centred frame
(`p_t = -dp_t`), `set_task` is ignored, the grid is a search window of
`window_margin` (6 m) around start and goal with no keep-out band, and the
ground height is estimated from rings of equal-height downward LiDAR returns.
This variant quantifies how much of Map-A*'s performance comes from the
declared task information; in development its additional failures were
flight-region exits (crashes) near the start.

Implementation: `baselines/astar/policy.py` (docstring describes the pipeline).
The shortest path is computed with A* on the cost grid (`astar_grid`,
8-connected, octile-distance heuristic; moving between neighbouring cells a and b
costs step length x (c_a + c_b) / 2, every cell cost is >= 1, so the heuristic is
admissible and the path is optimal). `cloud/test_astar.py` checks optimality
against a Dijkstra reference (scikit-image, used only by that test; skipped if
not installed). The planner itself needs only numpy/scipy.

## Environment version

All final results use the environment version of the thesis DreamerV3 results:
`drone_nav/envs/nav_aviary.py` **without** the per-episode PID reset (the fix
was evaluated and reverted, see project note `dreamer_reeval_pidreset_20261029`).
The frozen test uses `cloud/test_maps_diagnostics/`, the suite used for the
DreamerV3 test tables, so Map-A* and DreamerV3 share maps, environment and test
protocol (16 workers x 16 maps in fixed order).

Tuning (`tune_astar.py`) resets the PID controllers at the start of each
validation episode, so tuning is deterministic and independent of episode
order; with this, tuning results are identical whether or not the environment
contains the fix (checked on 10_dense).

## Setup (once)

```bash
python -m pytest -q cloud/test_astar.py
# optional, enables the A*-vs-Dijkstra optimality test:
# $CONDA_ROOT/envs/drone/bin/python -m pip install scikit-image
```

## 1. Validation tuning (cluster)

Same protocol and budget as the APF tuning: 32 candidates (defaults + 31
Latin-hypercube samples), successive halving 32 → 8 → 3 candidates on
32 / 64 / 128 validation maps, ranking by success, then collision+crash, then
time to goal. Validation seeds only.

```bash
sbatch --array=0-4 cloud/tune_astar_slurm.sh            # one task per scenario
python cloud/report_baseline_tuning.py /projects/EEHPC-DEV-2026D07-102/dreamer/logdir/baseline_tuning/astar
```

Each task writes `<scenario>/<scenario>_frozen_params.json`.

Variant without flight region (same budget, separate output root `astar_noregion`):

```bash
TASK_INFO=none sbatch --array=0-4 cloud/tune_astar_slurm.sh
python cloud/report_baseline_tuning.py /projects/EEHPC-DEV-2026D07-102/dreamer/logdir/baseline_tuning/astar_noregion
```

## 2. Frozen test (cluster, once, after tuning)

```bash
sbatch --array=0-4 cloud/eval_astar_test_slurm.sh
TASK_INFO=none sbatch --array=0-4 cloud/eval_astar_test_slurm.sh
```

Do not change parameters after looking at test results.

## Development note

Default parameters were set while debugging on about 30 validation maps
(seeds 170000–170007, 180000–180011, 190000–190005, 200000–200007,
210000–210005). The no-region variant was smoke-tested on 32 validation maps
(first 8 seeds of 10_sparse, 10_dense, 20_dense, 20_sparse_dynamic) without
changing any parameter. No test maps were used.
