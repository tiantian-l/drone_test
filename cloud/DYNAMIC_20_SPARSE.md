# 20_sparse + 10 个动态障碍物：A100 训练

## 最终环境

组合：`drone_nav drone_perc_cnn_hi drone_static_20_sparse drone_dynamic_20_sparse drone_reward_v2`。

- 20×20 m 障碍物区域；44 个静态箱体 + 10 个动态圆柱（density=0.025）。
- 动态速度 0.3–1.0 m/s，直径 0.25/0.50/0.75/1.00 m，高度 5 m。
- 每回合打乱直径列表后循环分配，数量为 3/3/2/2，但额外数量不总偏向小尺寸。
- 无人机速度上限 2 m/s；控制 30 Hz、物理 240 Hz；回合 40 s。
- 原 LiDAR 4 m、8×72，原 CNN-hi、原 reward-v2；不向策略暴露障碍物真实速度。
- 动态预设使用 `collision_reverse`：匀速直线，下一步扫掠区域若碰边界、静态障碍物或动态障碍物的保留区域，则本步不移动并反向。检查保守膨胀 AABB，避免薄墙穿透及动态交叉穿透。
- 这是运动学障碍物模型，不是弹性碰撞/行人模型；转向瞬时、障碍物不主动避让无人机。保守区域和固定处理顺序可能造成提前反向、短距离往返或局部停滞；看视频确认交互质量，不能仅凭成功率判断预测能力。
- 原 `drone_dynamic_density` 与 `boundary` 模式保留，不改变旧实验运动规律。生成不足或动态物体创建失败现在会显式报错。
- 静态 BFS 不保证时空可行性。评估地图种子 210000 起、16 workers×8 maps=128 episodes；每张地图固定一次动态初始化。多动态相位/独立地图随机流尚未实现。

## 文件与部署

主仓库新增 `cloud/run_dynamic_20_sparse.sh`、`cloud/train_dynamic_20_sparse_slurm.sh`、`cloud/resume_dynamic.py`、`cloud/check_dynamic_env.py`、`cloud/test_dynamic_obstacles.py`、`drone_nav/moving_geometry.py`，修改环境及适配器。

**配置还修改了 Git 子模块 `third_party/dreamerv3/dreamerv3/configs.yaml`。上传服务器时必须一起同步子模块修改；只上传主仓库文件不够。** 本次没有创建提交，也没有启动集群作业。

复用原 A100 Conda `drone` 环境；没有新增训练依赖。以下命令从服务器仓库根目录执行。Slurm 账户和分区沿用项目已有配置，可通过 sbatch 参数覆盖。

## 先验证，再训练

```bash
export CONDA_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3
export ENV_NAME=drone
export PYTHONPATH="$PWD:$PWD/third_party/dreamerv3:${PYTHONPATH:-}"

"$CONDA_ROOT/envs/$ENV_NAME/bin/python" -m unittest discover -s cloud -p test_dynamic_obstacles.py
"$CONDA_ROOT/envs/$ENV_NAME/bin/python" cloud/check_dynamic_env.py
```

第二条检查真实物理环境的 44+10 数量、8×72 观测、三个种子的 40 秒运动无重叠以及 reset 可复现性。Slurm 默认也会执行这个检查（`PREFLIGHT=0` 可跳过）。它不代表 GPU 学习链路已经通过；再做 200k 步校准任务（包含 100k 时的周期评估）：

```bash
MODE=scratch SEED=0 STEPS=200000 \
LOG_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_pilot \
sbatch cloud/train_dynamic_20_sparse_slurm.sh
```

正式推荐从已训练的 **CNN-hi + 20_sparse + reward-v2** 静态 agent checkpoint 初始化：

```bash
MODE=finetune SEED=0 STEPS=3000000 \
INIT_CHECKPOINT=/absolute/path/to/static_run/best_checkpoints/your_checkpoint.ckpt \
sbatch cloud/train_dynamic_20_sparse_slurm.sh
```

将 checkpoint 占位路径替换为真实路径。新动态目录必须为空；默认输出为 `/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_20_sparse_a100/seed_0`。

微调只通过 agent 加载接口读取源 checkpoint，动态环境步数从 0 开始，replay 新建；不复制静态 replay。agent 内部参数/优化器状态和计数器按现有 `agent.load` 恢复，因此不是“重置优化器的纯权重迁移”。架构和张量形状必须匹配。没有修改学习率或 imagination horizon。

从零训练的对照组使用独立目录：

```bash
MODE=scratch SEED=0 STEPS=5000000 \
LOG_ROOT=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_scratch_a100 \
sbatch cloud/train_dynamic_20_sparse_slurm.sh
```

续跑动态任务：

```bash
MODE=resume STEPS=3000000 \
LOGDIR=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_20_sparse_a100/seed_0 \
sbatch cloud/train_dynamic_20_sparse_slurm.sh
```

`STEPS` 是动态 run 的**总目标步数**，不是追加步数。resume 读取该目录保存的完整配置和 checkpoint/replay，不再读取静态初始化 checkpoint；不要删除 replay 文件。resume 模式不使用命令行附加配置参数。不可对同一 LOGDIR 同时提交两个任务。

续训时可通过环境变量 `RESUME_BATCH_SIZE=32` 显式覆盖 batch size；不设置时沿用保存值。`batch_length`、学习率和 `train_ratio` 不变。16→32 会让每次更新的数据量翻倍，在相同 train ratio 下更新次数约减半，并增加显存需求；不保证提速或改善收敛。加载后会重新编译，新 batch size 会写入该 run 的 config.yaml，后续续训沿用新值。此选项需要同步更新后的 `cloud/resume_dynamic.py`，尚未在 GPU 上验证。

每 300 秒保存 rolling checkpoint，墙钟到期后重提续跑；可能丢失最后一次保存后的最多约 5 分钟工作，I/O/评估可能延迟保存。当前训练循环正常结束也不保证立刻额外保存一次 rolling checkpoint；best checkpoints 按评估保存。不要把 Slurm 的 6 小时资源请求误认为 3M 步一定能跑完。

录像默认关闭以减少采样成本。需要检查行为可设置 `LOG_IMAGE=True`，worker 0 每 100 回合录像一次。若第一阶段太难，在单独实验目录传 `--env.drone.dynamic_obstacle_density 0.01` 做 4 个障碍物实验；不要修改主设置的基线含义。

## 模型、资源与训练时长

- 模型沿用 nominal `size12m` 预设，RSSM deter=2048、hidden=256、stoch=32、classes=16；CNN-hi 是专用 LiDAR 分支。名称不代表本任务实际精确参数数目，以训练启动打印为准。
- batch_size=16，batch_length=64（约 2.13 秒观测），imag_length=15（约 0.5 秒）；lr=4e-5；train_ratio=512。先保持这些参数以兼容静态模型、控制比较变量。
- 单张 A100，BF16；默认请求 80GB 分区、32 CPU、96GB 主存。40GB 可通过 `sbatch --partition=normal-a100-40 ...` 尝试，但本地没有测量显存峰值，不能保证容量。80GB 同样需要先完成 pilot。
- 16 个训练环境、16 个评估环境，评估配额 128 回合，每 100k 训练步评估一次。不要独立修改 eval_envs 而忘记 eval_seed_stride 和 eval_maps_per_env。
- replay 容量 5M；仅 576 维 float32 LiDAR 满容量约 11.5 GB（十进制），尚未计 state、动作、索引、评估 replay、缓存和运行时。96GB 是资源预算起点，不是实测峰值。
- A100 原生支持 BF16：https://www.nvidia.com/en-us/data-center/a100/ 。实际吞吐还取决于 PyBullet、CPU 配额、训练比例和评估开销。

没有可用的本项目 A100 动态训练日志，不能给出可信的固定耗时。用 pilot 中两个相隔至少一个评估周期的训练步数和墙钟时间计算 **有效训练环境步/秒**；不要直接用 `fps/train`（训练样本吞吐），`fps/policy` 也可能包含评估交互。

公式：剩余小时 = (目标步数 - 当前步数) / 有效环境步每秒 / 3600。

| 假设有效吞吐，含周期评估 | 3M 步 | 5M 步 |
|---|---:|---:|
| 100 steps/s | 8.3 h | 13.9 h |
| 200 steps/s | 4.2 h | 6.9 h |
| 400 steps/s | 2.1 h | 3.5 h |

这只是条件换算，不是 A100 实测。首次编译、初始化和排队另计。3M 环境步在 ratio=512、batch=16×64 下约对应 1.5M 次优化更新（忽略预填充与调度边界），训练计算量不小。

微调先预算 3M 新步，从零训练先预算 5M；都不是收敛保证。第一轮 200k 验证稳定性和吞吐，0.5M–1M 看动态碰撞是否下降，3M 后根据固定评估集趋势决定延长。最终建议 3 个训练种子；多 seed 共享同一静态 checkpoint 只反映动态阶段随机性，需要在报告中明确。

## 评估与日志

原 success/collision/crash/timeout 指标保留，增加 `static_collision`、`dynamic_collision`、`dynamic_count`（期望 10）。碰撞类别可能同时为真，不应简单相加视为互斥结果。它们是日志字段，不进入策略输入。top-k 阈值设为 0，使动态早期低成功率模型也能保存。

使用现有 `cloud/eval_checkpoint.py`，传动态 run 的 config.yaml 和对应 checkpoint，即可固定地图评估；正式报告应再准备独立测试种子，避免把选择 best checkpoint 的验证集当成测试集。

## 验证范围

本地已运行无第三方依赖的扫掠几何测试、Python 语法检查和 shell dry-run。当前本地缺少 NumPy/PyBullet/JAX 训练依赖，没有执行真实物理 preflight、checkpoint 加载、GPU 训练或性能测量。服务器第一次提交会先运行 preflight；checkpoint 兼容性和显存仍需 pilot 验证。
