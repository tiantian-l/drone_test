# 相同混合环境、随机初始化基线

## 比较目标

对照已有静态 checkpoint 微调实验，检验静态预训练对动态阶段样本效率和最终表现的影响。保持 44 个静态箱体 + 10 个动态圆柱，速度 0.3–1.0 m/s、直径 0.25/0.50/0.75/1.00 m、高度 5 m、collision_reverse 运动规则、40 秒回合。继续使用 CNN-hi（8×72 LiDAR）、reward-v2、nominal size12m 模型、学习率 4e-5、batch_length=64、imag_length=15、train_ratio=512。

新训练随机初始化网络和优化器，replay 为空。它不是“只有动态障碍物”的环境，也不是 random_agent：策略仍然会学习。源 checkpoint 和旧日志不会复制过来。

默认 batch_size=32，对应当前续训设置；原微调实验早期为 16，后期为 32，所以历史曲线不是严格的单因素初始化对照。正式归因需要两组都使用一致 batch 设置，或记录并对齐切换步数。若需要全程 16，用 BATCH_SIZE=16 并指定独立 LOG_ROOT，避免覆盖 batch32 实验。

## 首次提交（服务器仓库根目录）

新增脚本复用现有动态启动入口，不改变原微调脚本或环境代码。同步两个新脚本，并保留已修复 main guard 的 resume_dynamic.py。

```bash
MODE=scratch SEED=0 STEPS=3000000 BATCH_SIZE=32 LOG_IMAGE=True \
sbatch cloud/train_dynamic_scratch_slurm.sh
```

默认日志：

```text
/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_scratch_a100/seed_0
```

默认资源：1×A100 80GB、32 CPU、96GB 主存、BF16、6 小时。分区允许时可加 `sbatch --time=15:00:00`。脚本先执行物理环境检查。首次启动要求日志目录为空；继承的 INIT_CHECKPOINT 会被忽略，实际传入的 run.from_checkpoint 为空。MODE=finetune 被拒绝。

先短跑检查 GPU 和吞吐时，设置 STEPS=200000，并将 LOG_ROOT 改为独立 pilot 目录。录像默认开启，只由 worker 0 每 100 回合记录一次；想关闭可设置 LOG_IMAGE=False。

## 时间用完后的续训

```bash
MODE=resume STEPS=3000000 \
LOGDIR=/projects/EEHPC-DEV-2026D07-102/dreamer/logdir/drone_dynamic_scratch_a100/seed_0 \
sbatch cloud/train_dynamic_scratch_slurm.sh
```

STEPS 是总目标步数。不要重新用 scratch 启动已经有数据的目录，也不要两个任务同时写一个目录。resume 会恢复模型、优化器、步数和 replay，配置从该目录读取；BATCH_SIZE 和 LOG_IMAGE 在 resume 时不会覆盖保存值。RESUME_BATCH_SIZE 仍由公共续训入口支持，但基线比较期间建议不改变它。

延长到 5.6M 步时，将同一续训命令的 STEPS 改为 5600000。

## 多种子

先完成 seed0 的 pilot，再提交正式多种子实验。例如尚未启动任何正式 seed 时：

```bash
MODE=scratch STEPS=3000000 BATCH_SIZE=32 \
sbatch --array=0-2%1 cloud/train_dynamic_scratch_slurm.sh
```

不要设置或继承 SEED；数组编号自动作为种子，`%1` 限制同时只跑一个作业。若 seed0 已经启动，改为 `--array=1-2%1`，不要重复提交它。三个种子分别写入 seed_0/seed_1/seed_2。

## 预算与分析

先以相同的 3M 动态环境步比较成功率曲线、碰撞类别、达到预先选定成功率阈值的步数和最终表现。源静态模型曾训练 2.6M 步，因此微调 3M 的总样本成本约为 5.6M；可把从头组延长到 5.6M，增加总步数近似匹配的比较。不同 batch 调度和训练环境使总步数匹配不等于算力匹配，另报告 GPU 小时。

此前微调日志 fps/policy 约 39，仅用作粗预算量级：3M 步约 21 小时，5.6M 步约 40 小时，每个种子分别计费。这不是新基线实测，且 policy fps 可能计入评估动作，实际应使用训练步增量除以包含周期评估的墙钟时间。batch32、早期失败回合长度和重置频率都会改变吞吐。

评估配置保持 16 workers×8 maps=128 回合、每 100k 步一次、固定 eval seed base=210000。这些是选模验证图，不应同时作为论文唯一测试集；最终在独立测试种子上比较。前期随机策略成功率低、碰撞高是可能的，不应仅凭前 100k 曲线停止训练。

## 本地验证

启动命令回归测试覆盖：未加载继承的静态 checkpoint、batch/步数覆盖、独立 seed 目录、拒绝覆盖已有 run、拒绝 finetune。另运行了续训 spawn 导入回归测试和 shell 语法检查。未提交集群作业、未实测 GPU 显存和收敛情况。
