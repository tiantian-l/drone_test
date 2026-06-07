"""Embodied adapter that wraps a *gymnasium* environment.

DreamerV3 ships `embodied.envs.from_gym.FromGym`, but it targets the legacy
`gym` API (4-tuple `step`, single-value `reset`).  gym-pybullet-drones uses the
modern `gymnasium` API (5-tuple `step`, `(obs, info)` `reset`).  This adapter
bridges that gap and also forwards the `is_terminal` flag from `info` so that
DreamerV3 correctly distinguishes time-limit truncation from real terminations.
"""
import functools

import elements
import embodied
import numpy as np


class FromGymnasium(embodied.Env):

    def __init__(self, env, obs_key="state", act_key="action",
                 log_keys=(), log_image=False, image_key="log/image"):
        self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        # Extra scalar metrics pulled from `info` and exposed as `log/<key>`
        # so the DreamerV3 train loop aggregates them per episode (avg/max/sum).
        self._log_keys = tuple(log_keys)
        # Optional third-person RGB frames exposed as `log/image`, which the
        # train loop turns into a video for worker 0.
        self._log_image = bool(log_image)
        self._image_key = image_key
        self._done = True
        self._info = None

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        extra = {
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        for key in self._log_keys:
            extra[f"log/{key}"] = elements.Space(np.float32)
        if self._log_image:
            extra[self._image_key] = elements.Space(
                np.uint8, self._env.video_shape)
        return {**spaces, **extra}

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = elements.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            obs, self._info = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        obs, reward, terminated, truncated, self._info = self._env.step(action)
        self._done = bool(terminated) or bool(truncated)
        is_terminal = bool(self._info.get("is_terminal", terminated))
        return self._obs(
            obs, reward,
            is_last=self._done,
            is_terminal=is_terminal)

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal)
        # Inject per-step scalar metrics from info as log/<key>.
        info = self._info or {}
        for key in self._log_keys:
            obs[f"log/{key}"] = np.float32(float(info.get(key, 0.0)))
        # Inject the third-person video frame as log/image.
        if self._log_image:
            obs[self._image_key] = np.asarray(
                self._env.render_frame(), dtype=np.uint8)
        return obs

    def render(self):
        return self._env.render()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if hasattr(value, "spaces"):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return elements.Space(np.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)


def make_drone_nav(task, log_image=False, **kwargs):
    """Factory used by DreamerV3's `make_env` for the ``drone`` suite.

    ``task`` selects a preset; everything after the first ``_`` is the preset
    name (e.g. ``drone_nav`` -> task == "nav").  Extra kwargs come from
    ``config.env.drone`` and are forwarded to ``NavigationAviary``.

    Parameters
    ----------
    log_image : bool
        If True, render a third-person RGB frame each step and expose it as the
        ``log/image`` observation, which DreamerV3 logs as a video (worker 0).
        Adds rendering + replay cost, so keep it off for the bulk of training
        and enable it on a dedicated short run when you want to *see* behavior.
    """
    from drone_nav.envs.nav_aviary import NavigationAviary

    env = NavigationAviary(log_video=log_image, **kwargs)
    return FromGymnasium(
        env,
        obs_key="state",
        log_keys=("distance", "is_success"),
        log_image=log_image,
    )
