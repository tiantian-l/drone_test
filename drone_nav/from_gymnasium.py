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
                 log_keys=(), log_image=False, image_key="log/image",
                 worker_index=0, video_every=0):
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
        # Periodic rendering: to keep the cost negligible, only worker 0 ever
        # renders, and only once every `video_every` episodes. Every other env
        # (and every non-recording episode) emits a cheap zero placeholder of
        # the same shape so the observation space stays consistent across envs.
        self._worker_index = int(worker_index)
        self._video_every = int(video_every)
        self._episode = -1          # incremented to 0 on the first reset
        self._recording = False
        self._done = True
        self._info = None

    def _should_record(self):
        if not self._log_image or self._worker_index != 0:
            return False
        if self._video_every <= 0:
            return True  # record every episode (worker 0 only)
        return (self._episode % self._video_every) == 0

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
            extra["log/video_recorded"] = elements.Space(np.float32)
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
            self._episode += 1
            self._recording = self._should_record()
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
        # Inject the third-person video frame as log/image. Render only during
        # a recording episode (worker 0, every `video_every` episodes); emit a
        # zero placeholder otherwise so the shape is identical across all envs.
        if self._log_image:
            obs["log/video_recorded"] = np.float32(1.0 if self._recording else 0.0)
            if self._recording:
                frame = np.asarray(self._env.render_frame(), dtype=np.uint8)
            else:
                frame = np.zeros(self._env.video_shape, dtype=np.uint8)
            obs[self._image_key] = frame
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


def make_drone_nav(task, log_image=False, video_every=20, index=0, **kwargs):
    """Factory used by DreamerV3's `make_env` for the ``drone`` suite.

    ``task`` selects a preset; everything after the first ``_`` is the preset
    name (e.g. ``drone_nav`` -> task == "nav").  Extra kwargs come from
    ``config.env.drone`` and are forwarded to ``NavigationAviary``.

    Parameters
    ----------
    log_image : bool
        If True, expose a ``log/image`` observation that DreamerV3 logs as a
        video. The key is present on *every* env (so the spaces stay
        consistent), but frames are only rendered on worker 0 during recording
        episodes; all other steps emit a cheap zero placeholder.
    video_every : int
        Render one episode every ``video_every`` episodes (worker 0 only).
        Larger -> cheaper. Set to 0 to record every episode on worker 0.
    index : int
        The parallel env/worker index supplied by DreamerV3's ``make_env``.
        Only worker 0 renders, so the bulk of envs pay no rendering cost.
    """
    from drone_nav.envs.nav_aviary import NavigationAviary

    # Only worker 0 ever needs to render frames; let the rest skip the PyBullet
    # camera setup entirely.
    env = NavigationAviary(log_video=bool(log_image) and index == 0, **kwargs)
    return FromGymnasium(
        env,
        obs_key="state",
        log_keys=("distance", "is_success"),
        log_image=log_image,
        worker_index=index,
        video_every=video_every,
    )
