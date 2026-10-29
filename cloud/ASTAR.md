# Map-A* reference baseline

A map-based classical baseline: online LiDAR occupancy grid + grid shortest-path
replanning + pure-pursuit tracking + a local LiDAR safety layer. It complements
the reactive APF baselines (branch `feature/apf`) with explicit memory and
global planning. This branch is self-contained: it does not import APF or PPO
code.

## Information available to the planner

- The same observation as the DreamerV3 policies (goal displacement, velocity,
  attitude, 8×72 LiDAR).
- **Declared task information** (option A): the goal position in the world
  frame, given once per episode via `set_task`, and the flight region from the
  environment settings. This lets the planner place the flight-region boundary
  in its map. Report this in the thesis.
- No simulator obstacle geometry and no obstacle velocities.

Implementation: `baselines/astar/policy.py` (docstring describes the pipeline).
The shortest path is computed with Dijkstra's algorithm on the cost grid
(`skimage.graph.MCP_Geometric`); A* with an admissible heuristic returns a path
of the same cost.

## Environment version

`drone_nav/envs/nav_aviary.py` on this branch is byte-identical to
`feature/apf` (includes the per-episode PID reset). The test manifests in
`cloud/test_maps_pidreset_v2/` are the ones used for the APF test run and
validate against these sources, so APF and Map-A* are tested on the same maps
and environment version.

## Setup (once)

```bash
$CONDA_ROOT/envs/drone/bin/python -m pip install scikit-image
python -m pytest -q cloud/test_astar.py
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

## 2. Frozen test (cluster, once, after tuning)

```bash
sbatch --array=0-4 cloud/eval_astar_test_slurm.sh
```

Do not change parameters after looking at test results. Run the DreamerV3
re-evaluation and, if possible, the APF/APF-Tangent test on the same cluster so
that all methods share machine and environment version.

## Development note

Default parameters were set while debugging on about 30 validation maps
(seeds 170000–170007, 180000–180011, 190000–190005, 200000–200007,
210000–210005). No test maps were used.
