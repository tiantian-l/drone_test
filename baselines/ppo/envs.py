"""Gymnasium factories; no Embodied adapter or JAX imports."""
import copy
import inspect
from functools import partial

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .config import FRAMEWORK_KEYS


class EpisodeGeometry(gym.Wrapper):
    """Capture geometry before motion and attach it to the terminal info."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        env = self.unwrapped
        self.geometry = dict(seed=int(env.CURRENT_EVAL_SEED),
                             start=env.INIT_XYZS[0].tolist(), goal=env.TARGET_POS.tolist(),
                             static_boxes=[list(map(float, x)) for x in env._obstacle_specs],
                             dynamic_cylinders=[list(map(float, x)) for x in env._dynamic_obstacle_specs])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info, initial_geometry=self.geometry)
        return obs, reward, terminated, truncated, info


class FirstResetSeed(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.initial_seed = seed

    def reset(self, *, seed=None, options=None):
        if self.initial_seed is not None:
            # SB3 seeds VecEnv on model creation using seed+index. Preserve the
            # documented SeedSequence stream shared with the Dreamer factory.
            seed = self.initial_seed
            self.initial_seed = None
        return self.env.reset(seed=seed, options=options)


def make_env(settings, index, seed=0, evaluation=False, quota=8, stride=16):
    from drone_nav.envs.nav_aviary import NavigationAviary
    kwargs = {k: copy.deepcopy(v) for k, v in settings.items() if k not in FRAMEWORK_KEYS}
    unknown = set(kwargs) - set(inspect.signature(NavigationAviary.__init__).parameters)
    if unknown:
        raise ValueError(f'Unknown navigation settings: {sorted(unknown)}')
    kwargs.update(gui=False, log_video=False, eval_mode=evaluation,
                  eval_seed=int(settings['eval_seed_base']) + index if evaluation else 0,
                  eval_maps_per_env=quota, eval_seed_stride=stride)
    env = NavigationAviary(**kwargs)
    if evaluation:
        env = EpisodeGeometry(env)
    else:
        worker_seed = int(np.random.SeedSequence([seed, index]).generate_state(1, dtype=np.uint32)[0])
        env = FirstResetSeed(env, worker_seed)
    return Monitor(env)


def vector_env(settings, count, seed=0, evaluation=False, quota=8, stride=16, subprocess=True):
    factories = [partial(make_env, settings, i, seed, evaluation, quota, stride) for i in range(count)]
    if subprocess and count > 1:
        return SubprocVecEnv(factories, start_method='spawn')
    return DummyVecEnv(factories)
