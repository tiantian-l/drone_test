# Drone navigation A–D feasibility ablation

## Objective

The experiment identifies the first task scale at which learning breaks down.
A–D use the same 12M DreamerV3 model, optimizer, replay settings, control rate,
training ratio (512), and eight training environments. The current feasibility
pass uses one training seed and increasing step ceilings to avoid spending the
full D budget on tasks that were already shown to work. Learning curves must be
compared at common step checkpoints when making a strict sample-efficiency
claim.

| Variant | Approx. route | Obstacles | Episode | Step ceiling | Capability tested |
|---|---:|---:|---:|---:|---|
| A | 4–6 m | 0 | 15 s | 0.5M | flight and goal reaching |
| B | 8–10 m | 16 | 20 s | 1M | local obstacle avoidance |
| C | 14–16 m | 40 | 30 s | 2.5M | medium-range planning |
| D | ~24 m | 88 | 40 s | 5M | full target task |

Each evaluation uses 16 workers with four deterministic map slots each, for 64
unique maps and 64 counted episodes. One seed is enough
for this feasibility pass. Once the final method and its key baseline have been
selected, add independent seeds and report mean and standard deviation for the
formal result.

## Running

On each school computer, install the environment once from the repository root:

```bash
bash cloud/setup_school.sh
```

The training scripts then load Conda and activate the `drone` environment
automatically, just like `cloud/train_school.sh`; manual `conda activate` is not
required. The defaults are `$HOME/miniconda3` and environment name `drone`.
Override them with `CONDA_ROOT` and `ENV_NAME` if necessary.

The general A–D launcher is:

```bash
bash cloud/run_ablation.sh
```

For the two-computer, single-seed feasibility study, use the dedicated scripts:

```bash
# Computer 1: A (0.5M steps), then B (1M steps).
bash cloud/run_ablation_ab.sh

# Computer 2: C (2.5M steps), then D (5M steps).
bash cloud/run_ablation_cd.sh
```

Both default to seed 0 and write the same directory layout below
`~/logdir/drone_ablation`, so their result folders can later be copied under one
common root and analyzed together. The two computers may run simultaneously; A
and B are sequential on the first computer, while C and D are sequential on the
second computer.

Low-frequency policy videos are enabled by default and rendered only by worker
0. A/B record one episode every 50 episodes; C/D record one every 100 episodes.
Override the interval or disable video entirely with:

```bash
VIDEO_EVERY=200 bash cloud/run_ablation_cd.sh
LOG_IMAGE=False bash cloud/run_ablation_ab.sh
```

Budgets and runtime settings can be overridden without editing the scripts:

```bash
A_STEPS=300000 B_STEPS=600000 SEED=0 bash cloud/run_ablation_ab.sh
C_STEPS=1500000 D_STEPS=3000000 SEED=0 bash cloud/run_ablation_cd.sh

# Use CPU or choose a different output root when needed.
JAX_PLATFORM=cpu LOG_ROOT=/path/to/logs bash cloud/run_ablation_ab.sh
```

Useful overrides:

```bash
# Check generated commands without training.
DRY_RUN=1 bash cloud/run_ablation.sh

# Run a subset or a quick pipeline check.
VARIANTS=a,b SEEDS=0 bash cloud/run_ablation.sh --run.steps 10000

# CPU debugging only (full training is intended for CUDA).
VARIANTS=a SEEDS=0 JAX_PLATFORM=cpu bash cloud/run_ablation.sh --run.steps 1000
```

Runs are written to `~/logdir/drone_ablation/{A,B,C,D}/seed_N` by default.
Existing non-empty run directories are resumed by DreamerV3 checkpoints, so use
a new `LOG_ROOT` when starting a genuinely new experiment series.

### Dynamic-obstacle D task

The optional `drone_dynamic_density` preset keeps D's 20 x 20 m obstacle field
and 88 static boxes, then adds 20 moving cylinders (0.05/m²), matching the
reference scene's dynamic-obstacle density. Diameters are 0.25, 0.50, 0.75,
and 1.00 m (five each), height is 5.0 m, speed is sampled from 0.3--1.0 m/s,
and the episode limit is extended to 60 seconds.

```bash
python third_party/dreamerv3/dreamerv3/main.py \
  --configs drone_nav drone_ablation_d drone_dynamic_density
```

Dynamic cylinders are excluded from the static BFS feasibility check, but are
real collision bodies seen by LiDAR and used by clearance shaping and collision
termination. Their deterministic straight-line motion reflects at map bounds.

Summarize completed runs with:

```bash
python3 cloud/analyze_ablation.py ~/logdir/drone_ablation --threshold 0.8
```

The command prints a Markdown table and writes
`~/logdir/drone_ablation/summary.csv`.
The primary metric is deterministic-evaluation success. Collision, crash,
timeout, final distance, and the environment step at which success first reaches
80% explain why a variant fails and measure sample efficiency.

## Interpretation and training-parameter decisions

Use the first failed transition to select the next intervention:

- **A fails:** do not increase model size. Validate action parameterization,
  reward/termination semantics, and the Gymnasium adapter first.
- **A succeeds but B fails:** the problem is local avoidance. Test a longer
  lidar range and a learnable signed speed mapping; keep the route short.
- **B succeeds but C fails:** the likely limit is temporal credit assignment.
  First test `agent.imag_length: 30` and `batch_length: 96`. This increases GPU
  cost, so it belongs in a second-stage training ablation, not the A–D baseline.
- **C succeeds but D fails:** introduce a B→C→D curriculum or initialize D from
  a C checkpoint. Compare that against D from scratch with the same total number
  of environment steps.
- **D succeeds inconsistently across seeds:** retain the environment and increase
  seeds/evaluation maps before changing the agent.

Do not compare variants only by final score: easier tasks may converge much
earlier. Plot success against environment steps and require declining collision
or timeout rates. A useful feasibility criterion is evaluation success ≥80% on
at least two of three seeds, with no seed dominated by crashes.

After locating the failure boundary, change one training factor at a time. A
recommended second-stage order is: action mapping, reward balance, imagination
length, sequence length, then model size. Changing all of them together would
make it impossible to identify what restored learning.

## Seed scope and reproducibility

The current feasibility pass trains one independent model per variant using
`seed=0`. Start, goal, and obstacle layouts are still randomized on every
training episode; a single training seed does not mean that the model sees only
one map or one initial state.

One seed is sufficient for locating the A–D failure boundary, but it does not
demonstrate training stability. After selecting the final method and its key
baseline, run additional independent seeds and report mean and standard
deviation for the formal thesis result.

Evaluation uses fixed maps derived from each variant's `eval_seed_base`. Worker
`i` and map slot `j` use `eval_seed_base + i + 16*j`, so task B evaluates seeds
120000--120063 exactly once per cycle. The driver requires four counted
episodes from every worker and discards surplus episodes from fast workers.
Every cycle is validated for 64 unique seeds before its metrics are accepted,
and map-level outcomes are appended to `eval_maps.jsonl`. Checkpoints from
different training runs are therefore compared on the same equally weighted
maps.

Inspect the latest map-level evaluation and list its failed maps with:

```bash
python3 cloud/analyze_eval_maps.py \
  ~/logdir/drone_ablation/B/seed_0/eval_maps.jsonl
```

## Reward baseline and follow-up

The first C run keeps the current reward unchanged. It combines progress,
goal, time, collision, crash, timeout, and obstacle-proximity terms. Keeping it
fixed provides a baseline and avoids changing the environment and reward at the
same time.

The current weights make progress comparatively strong: moving 10 m toward the
goal yields about +80 progress reward, whereas a later collision costs only
-10. In C or D this can favor a locally useful but unsafe strategy that flies
directly toward the goal and collides later. The progress scale also grows with
route length, so raw episode scores are not comparable across A–D; use success,
collision, timeout, and final-distance metrics instead.

Only introduce a Reward V2 if C shows decreasing target distance together with
a persistently high collision rate. The proposed follow-up should:

- normalize progress by the episode's initial goal distance so A–D have similar
  cumulative progress scales;
- make collision and crash penalties large enough to offset a failed trajectory;
- expand the obstacle safety margin from 0.5 m to approximately 0.8–1.0 m;
- keep model and other training parameters fixed during the reward comparison.

If C instead times out while remaining far from the goal, prioritize exploration,
temporal modeling, or curriculum design rather than increasing collision
penalties. If it crashes even without nearby obstacles, inspect the action
mapping and flight controller first.

## Blackwell GPU setup

RTX Blackwell GPUs with compute capability 12.0 require CUDA 12.8 or newer for
native compiler support. The project keeps JAX 0.4.33 for DreamerV3 compatibility
but installs `nvidia-cuda-nvcc-cu12>=12.8,<13`, so XLA uses a Blackwell-capable
`ptxas`. `cloud/setup_school.sh` now performs a real compiled matrix operation;
merely listing `jax.devices()` is not considered a sufficient GPU test.

If an existing environment reports `ptxas too old`, pull the updated repository
and rerun the idempotent setup script:

```bash
bash cloud/setup_school.sh
```

Verify the compiler and JAX execution before restarting a long run:

```bash
$HOME/miniconda3/envs/drone/bin/python -m pip show nvidia-cuda-nvcc-cu12
$HOME/miniconda3/envs/drone/bin/python -c \
  'import jax.numpy as j; print((j.ones((32,32)) @ j.ones((32,32))).block_until_ready())'
```

With JAX 0.4.33, Blackwell can still hit an LLVM failure while compiling mixed
precision (`Unsupported conversion from bf16 to f16`). The ablation launcher
therefore detects compute capability 10.x/12.x and selects `float32`
automatically; older CUDA GPUs retain `bfloat16`. This is a compiler
compatibility workaround, not an indication that Blackwell lacks BF16 hardware.

The selected dtype is printed before each run. It can be overridden explicitly:

```bash
JAX_COMPUTE_DTYPE=float32 bash cloud/run_ablation_ab.sh
JAX_COMPUTE_DTYPE=bfloat16 bash cloud/run_ablation_cd.sh
```

Do not force `bfloat16` on the affected Blackwell/JAX 0.4.33 combination. FP32
uses more GPU memory and may train more slowly, but the 12M model is the safer
first compatibility target. A later, separate dependency upgrade can test a
newer JAX CUDA 12.8 build before mixed precision is restored.
