# 最终独立测试评估 / Final held-out evaluation

**五个场景，每个 256 张地图，共 1280 张。**

**Five scenarios, 256 maps each, 1280 maps in total.**

## 1. 测试集定义 / Test suite definition

`cloud/test_maps_v1/*.json` 保存固定的**地图生成清单**，并非提前计算好的物理几何。每份清单包含 256 个地图种子、场景预设、评估进程分配，以及地图生成与配置源码的 SHA-256 指纹。现有环境按这些信息确定性地生成实际地图；障碍物运动前的初始几何会导出至 `maps.jsonl` 和单独的 `maps/*.json` 文件。使用 `--maps-only` 可以先生成地图，再评估策略。

`cloud/test_maps_v1/*.json` contains fixed **map recipes**, not precomputed physics geometry. Each manifest stores 256 map seeds, scenario presets, worker allocation, and SHA-256 fingerprints of the map/configuration source. The existing environment generates the geometry deterministically and exports its initial state, before obstacle movement, to `maps.jsonl` and individual `maps/*.json` files. Use `--maps-only` to materialize maps before evaluating policies.

| 场景参数 / Scenario argument | 请求的静态箱体数 / Requested static boxes | 动态圆柱数 / Moving cylinders | 测试种子 / Test seeds |
| --- | ---: | ---: | --- |
| `10_sparse` | 11 | 0 | 1000000–1000255 |
| `10_dense` | 22 | 0 | 1010000–1010255 |
| `20_sparse` | 44 | 0 | 1020000–1020255 |
| `20_dense` | 88 | 0 | 1030000–1030255 |
| `20_sparse_dynamic` | 44 | 10 | 1040000–1040255 |

这些种子范围互不重叠，也与当前 validation 集合分离：五个 validation 集合分别从 170000、180000、190000、200000、210000 开始，各有 128 张地图。评估程序还会检查测试种子是否与传入训练配置中保存的 validation 种子重叠。训练地图来自持续采样的随机流，因此这里**不能保证与历史上每一张训练地图都不同**。

These ranges are mutually disjoint and separate from the current validation sets: 128 maps each, starting at 170000, 180000, 190000, 200000, and 210000. The evaluator also checks for overlap with the validation schedule saved in the supplied run configuration. Training maps come from an ongoing random stream, so this **does not prove exclusion from every historical training layout**.

静态地图生成保留现有 BFS 可行性检查：若找不到可行布局，回退机制可能移除部分箱体，实际数量以导出的地图为准。动态场景沿用 `drone_dynamic_20_sparse`：10 个圆柱、密度 0.025、速度 0.3–1.0 m/s、`collision_reverse` 运动模式、回合上限 40 秒。静态 10 m 场景上限为 20 秒，20 m 场景为 40 秒。每张动态地图只有一组固定的初始位置和速度，并未测试多个运动相位。

Static generation retains the existing BFS feasibility fallback, which may remove boxes when no feasible layout is found; exported maps show the actual counts. The dynamic scene uses `drone_dynamic_20_sparse`: 10 cylinders, density 0.025, speed 0.3–1.0 m/s, `collision_reverse` motion, and a 40-second episode limit. Static 10 m tasks use 20 seconds; 20 m tasks use 40 seconds. Each dynamic map has one fixed initial position/velocity realization, not multiple motion phases.

## 2. 提交评估 / Submit evaluation

同步主仓库、五份地图清单，**以及修改过的 Dreamer 子模块文件**到集群。复用已有 Conda `drone` 环境，不需要添加训练依赖之外的软件包。在仓库根目录执行以下命令，将示例绝对路径替换为实际路径。

Sync the repository, five manifests, **and modified Dreamer submodule files** to the cluster. Reuse the existing Conda `drone` environment; no packages beyond the training dependencies are required. Run from the repository root, replacing the example absolute paths with real paths.

```bash
sbatch cloud/eval_final_slurm.sh 10_dense \
  /absolute/path/to/10_dense/seed_0 \
  /absolute/path/to/selected_checkpoint.ckpt
```

| 参数位置 / Position | 含义 / Meaning |
| --- | --- |
| 1 | 场景名称，例如 `10_dense`。 / Scenario name, e.g. `10_dense`. |
| 2 | 包含 `config.yaml` 的训练目录，用于恢复策略结构和感知配置。 / Training directory containing `config.yaml`, which supplies policy architecture and perception settings. |
| 3 | 选定的 checkpoint 路径。支持仅保存 agent 的最佳模型、完整 checkpoint 代目录，或滚动 `ckpt/` 目录。 / Selected checkpoint: an agent-only best checkpoint, a complete checkpoint generation directory, or a rolling `ckpt/` directory. |
| 4（可选 / optional） | 新的结果输出目录；若目录已存在则拒绝运行。 / New output directory; existing directories are rejected. |

滚动 `ckpt/` 的具体保存代在启动时解析一次。论文最终评估建议使用已经固定、选定的 checkpoint；加载期间不要修改或删除它。

A rolling `ckpt/` path is resolved to a checkpoint generation once at launch. For final paper results, prefer a frozen selected checkpoint; do not modify or delete it while loading.

评估其他场景时，修改第一个参数和对应的策略路径：`10_sparse`、`20_sparse`、`20_dense`、`20_sparse_dynamic`。也可以在动态场景中评估静态 `20_sparse` 策略，衡量迁移表现。测试场景会覆盖几何、动态障碍物及标准任务物理设置；模型结构和感知配置沿用传入的训练配置。checkpoint 与配置的参数形状不兼容时，加载会失败。

For other scenarios, change the first argument and corresponding policy paths: `10_sparse`, `20_sparse`, `20_dense`, or `20_sparse_dynamic`. The dynamic scene can also evaluate a static `20_sparse` policy to measure transfer. The test scenario overrides geometry, dynamics, and standard task physics; model architecture and perception remain from the supplied training configuration. Incompatible checkpoint/configuration shapes fail on load.

默认资源：1 张 A100 40GB、32 个 CPU、4 小时时限、BF16。评估使用 16 个新建环境进程，每个运行 16 张地图，共享 `freshenv_rng0_v2` 协议。GPU 运行流程尚未在本地验证。每次提交默认写入独立目录：

Default resources: one A100 40GB, 32 CPUs, a four-hour limit, and BF16. Evaluation uses 16 fresh workers with 16 maps each under the shared `freshenv_rng0_v2` protocol. The GPU runtime has not been validated locally. Each submission defaults to a separate output directory:

```text
/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/final_eval/SCENARIO/job_JOBID/
```

原训练日志和 checkpoint 只读。比较结果时，请保持硬件、依赖版本、计算精度、评估进程数和策略种子配置一致。不要单独修改进程数，因为它属于该协议的策略随机数调度规则。

Source training logs and checkpoints are read-only. Keep hardware, dependency versions, precision, worker count, and policy seed configuration fixed when comparing results. Do not change worker count independently: it is part of this protocol's policy RNG schedule.

## 3. 只生成地图，不运行策略 / Materialize maps without running the policy

在配置好的训练环境中执行。此模式只运行 CPU 物理环境，不创建 GPU agent。

Run in the configured training environment. This mode uses CPU physics and does not create a GPU agent.

```bash
export PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:$PWD/third_party/gym-pybullet-drones:${PYTHONPATH:-}"
python cloud/eval_final.py \
  --config /absolute/path/to/run/config.yaml \
  --manifest cloud/test_maps_v1/20_sparse_dynamic.json \
  --maps-only --output /absolute/path/to/new/map_export
```

对五份清单分别执行，即可导出全部场景。导出内容包括初始起点、目标点、静态箱体 `(x,y,width_x,width_y,height)`，以及动态圆柱 `(x,y,radius,height,vx,vy)`。

Repeat for each of the five manifests to export all scenarios. Exports include the initial start and goal, static boxes `(x,y,width_x,width_y,height)`, and dynamic cylinders `(x,y,radius,height,vx,vy)`.

评估时会根据同一清单重新生成地图，**不会直接读取已导出的几何文件**。可比较不同策略结果中的 `geometry_sha256`，确认实际地图一致。修改环境后不要直接重新生成清单来绕过检查；源码指纹不匹配会报错。如果确实需要定义新版本测试集，请生成到另一个目录：

Evaluation regenerates maps from the same recipes; it **does not load the exported geometry files directly**. Compare `geometry_sha256` across policy runs to confirm identical geometry. Do not regenerate manifests merely to bypass checks after changing the environment: source fingerprint mismatches are errors. To deliberately define a new test suite version, generate into a different directory:

```bash
python cloud/final_test_suite.py --output /absolute/path/to/new_suite
```

Slurm 提交时，通过环境变量 `TEST_SUITE_DIR` 指定新的清单目录。

For Slurm submissions, select the new manifest directory with the `TEST_SUITE_DIR` environment variable.

## 4. 输出文件与结果解释 / Outputs and interpretation

| 文件 / Files | 内容 / Contents |
| --- | --- |
| `episodes.jsonl`, `episodes.csv` | 每张地图一行：成功、碰撞、坠毁、超时、静态/动态碰撞标记、距离、回合长度及随机数审计字段。 / One row per map: success, collision, crash, timeout, static/dynamic collision flags, distances, episode length, and RNG audit fields. |
| `summary.json`, `summary.csv` | 比率和计数、Wilson 95% 置信区间、平均最终距离、平均回合长度，以及**仅成功回合的平均到达时间**。没有成功回合时，时间为 null，不是 0。 / Rates and counts, Wilson 95% intervals, mean final distance, mean episode length, and mean arrival time **among successful episodes only**. With no successes, time is null, not zero. |
| `maps.jsonl`, `maps/` | 全部 256 张地图的实际初始几何。重复几何或不完整/重复的种子、进程、地图槽位覆盖会报错。 / Actual initial geometry for all 256 maps. Duplicate geometry or incomplete/duplicate seed, worker, or slot coverage is an error. |
| `manifest.json`, `config.yaml` | 本次测试协议及实际生效的配置。 / Test protocol and effective configuration. |

汇总还记录 checkpoint 路径和哈希、精度、评估协议、地图及清单哈希、软件包版本和 Git 版本号。

The summary also records the checkpoint path and hash, precision, evaluation protocol, geometry and manifest hashes, package versions, and Git revision.

置信区间描述单个训练策略在地图样本上的不确定性，不能衡量不同训练 seed 之间的差异。碰撞、坠毁和成功标记沿用现有环境定义，在边界情况下不一定互斥。**必须用 validation 选择 checkpoint，不能按最终 test 分数选模型。** 此流程不会更新模型、加载 replay 或保存训练 checkpoint。

Confidence intervals describe map sampling uncertainty for one trained policy; they do not measure variability across training seeds. Collision, crash, and success flags follow the existing environment and may overlap in edge cases. **Select checkpoints using validation only, never final-test scores.** This workflow performs no training updates, replay loading, or training checkpoint saves.

## 5. 本地检查与集群验证 / Local checks and cluster validation

```bash
python -m unittest discover -s cloud -p test_final_test_suite.py
python -m unittest discover -s cloud -p test_evaluation_protocol.py
bash -n cloud/eval_final_slurm.sh
```

这些检查覆盖清单隔离性和完整性、地图覆盖、统计分母、checkpoint 路径解析，以及初始几何记录。它们不能替代集群上的 PyBullet 地图生成和真实 checkpoint 评估。建议先为每个场景生成实际地图，再评估选定的 checkpoint；使用相同设置重复运行，比较地图几何哈希和逐地图结果。

These checks cover manifest isolation and integrity, map coverage, summary denominators, checkpoint path resolution, and initial geometry recording. They do not replace PyBullet map generation and real checkpoint evaluation on the cluster. First materialize each scenario, then evaluate the selected checkpoint; repeat with the same setup and compare geometry hashes and per-map outcomes.

## 6. 结果文件命名 / Result file naming

结果目录简化为 `final_eval/<评估场景>/job_<作业编号>/`。来源策略和 checkpoint 标识直接写入文件名，格式为 `<POLICY_LABEL>__<CHECKPOINT_NAME>__<文件类型>`。上文的 `summary.csv`、`episodes.jsonl` 等是文件类型简称，实际文件均带此前缀；逐地图 JSON 子目录也带相同前缀。

Output directories use `final_eval/<evaluation_scene>/job_<job_id>/`. Source policy and checkpoint identity appear directly in filenames: `<POLICY_LABEL>__<CHECKPOINT_NAME>__<artifact>`. Names such as `summary.csv` and `episodes.jsonl` above refer to artifact types; actual filenames and the per-map JSON subdirectory carry this prefix.

例如 `20_sparse` 的 seed 0、220 万步 checkpoint 在 `10_dense` 上评估：

Example: evaluating the 2.2M-step checkpoint from `20_sparse`, seed 0, on `10_dense`:

```text
final_eval/10_dense/job_12345/
  20_sparse_seed_0__step_0002200000_success_0.9531__summary.csv
  20_sparse_seed_0__step_0002200000_success_0.9531__summary.json
  20_sparse_seed_0__step_0002200000_success_0.9531__episodes.csv
  20_sparse_seed_0__step_0002200000_success_0.9531__episodes.jsonl
  20_sparse_seed_0__step_0002200000_success_0.9531__maps.jsonl
  20_sparse_seed_0__step_0002200000_success_0.9531__maps/
  20_sparse_seed_0__step_0002200000_success_0.9531__config.yaml
  20_sparse_seed_0__step_0002200000_success_0.9531__manifest.json
```

`POLICY_LABEL` 默认从训练目录末两级名称推断，例如 `20_sparse_seed_0`。可以设置 `export POLICY_LABEL=4090_20_sparse_seed_0` 来标明硬件或实验版本。标签只影响命名和记录，不改变场景。checkpoint 使用解析后的名称，去掉 `.ckpt` 后缀；无 checkpoint 的地图生成模式使用 `maps_only`。不安全的文件名字符会替换为下划线，超长前缀会缩短并附加哈希。

`POLICY_LABEL` defaults to the last two components of the training directory, e.g. `20_sparse_seed_0`. Set `export POLICY_LABEL=4090_20_sparse_seed_0` to include hardware or experiment identity. The label affects naming and metadata only. The resolved checkpoint name is used without `.ckpt`; map-only mode uses `maps_only`. Unsafe filename characters are replaced with underscores; long prefixes are shortened with a hash suffix.

汇总仍保存 `scenario`、`policy_label`、`source_config`、`checkpoint` 和 `checkpoint_sha256`。已有输出不会被移动或重命名。第四个输出目录参数仍可覆盖默认目录。

Summaries retain `scenario`, `policy_label`, `source_config`, `checkpoint`, and `checkpoint_sha256`. Existing results are not moved or renamed. The fourth output-directory argument still overrides the default location.
