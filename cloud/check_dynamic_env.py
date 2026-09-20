"""CPU preflight using actual PyBullet bodies and the training configuration.

PYTHONPATH=.:third_party/dreamerv3:third_party/gym-pybullet-drones python cloud/check_dynamic_env.py
"""
from pathlib import Path
import numpy as np
import pybullet as p
import elements
from ruamel import yaml
from dreamerv3.main import make_env

presets = yaml.YAML(typ="safe").load((Path(__file__).resolve().parents[1] /
    "third_party/dreamerv3/dreamerv3/configs.yaml").read_text())
config = elements.Config(presets["defaults"])
for name in ("drone_nav", "drone_perc_cnn_hi", "drone_static_20_sparse",
             "drone_dynamic_20_sparse", "drone_reward_v2"):
    config = config.update(presets[name])
assert config.env.drone.n_obstacles == 44
assert config.env.drone.dynamic_obstacle_density == .025
assert config.env.drone.episode_len_sec == 40
wrapped = make_env(config, 0, eval_mode=True)
# Access the direct physics environment separately to test its collision shapes.
from drone_nav.envs.nav_aviary import NavigationAviary
kw = dict(config.env.drone)
for key in ("eval_seed_base", "eval_maps_per_env", "eval_seed_stride", "log_image",
            "video_every", "use_index"):
    kw.pop(key, None)
env = NavigationAviary(**kw)
try:
    for seed in (7, 19, 31):
        env.reset(seed=seed)
        initial = np.asarray(env._dynamic_obstacle_specs).copy()
        assert len(env._dynamic_obstacle_ids) == 10
        assert len(env._static_obstacle_ids) == 44
        assert env.observation_space["lidar"].shape == (8, 72)
        for step in range(1200):
            env._advance_dynamic_obstacles(1 / 30)
            if step % 10 == 0:
                for i, bid in enumerate(env._dynamic_obstacle_ids):
                    peers = env._static_obstacle_ids + env._dynamic_obstacle_ids[i + 1:]
                    for other in peers:
                        points = p.getClosestPoints(bid, other, distance=.049,
                                                   physicsClientId=env.CLIENT)
                        assert not points, (seed, step, bid, other, points)
        env.reset(seed=seed)
        np.testing.assert_array_equal(initial, env._dynamic_obstacle_specs)
    print("PASS: configuration, 44+10 bodies, 8x72 LiDAR, 3x40s non-overlap, deterministic reset")
finally:
    env.close()
    wrapped.close()
