"""A->B point navigation environment for a single quadrotor.

Built on top of gym-pybullet-drones `BaseRLAviary` using the *velocity*
controller (`ActionType.VEL`).  The observation is purely proprioceptive
(state-only), as requested:

    obs = [
        rel_x, rel_y, rel_z,     # goal_pos - drone_pos      (3)
        vx, vy, vz,              # linear velocity (world)   (3)
        roll, pitch, yaw,        # attitude (euler)          (3)
        wx, wy, wz,              # body angular velocity      (3, optional)
    ]

Action (4-dim, ActionType.VEL):
    a = [dx, dy, dz, speed]
    * (dx, dy, dz) is a (un-normalized) desired velocity direction, the
      DSL-PID velocity controller normalizes it to a unit vector.
    * speed in [-1, 1] -> |speed| scales the target speed up to SPEED_LIMIT.
"""
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class NavigationAviary(BaseRLAviary):
    """Single-agent A->B navigation with a velocity controller."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False,
                 # ---- task / curriculum knobs -------------------------------
                 goal_pos=None,
                 start_pos=None,
                 randomize_goal: bool = True,
                 randomize_start: bool = False,
                 goal_sample_range=((-2.0, 2.0), (-2.0, 2.0), (0.5, 2.0)),
                 start_sample_range=((-0.2, 0.2), (-0.2, 0.2), (0.9, 1.1)),
                 goal_tolerance: float = 0.10,
                 episode_len_sec: int = 12,
                 bounds=((-3.0, 3.0), (-3.0, 3.0), (0.05, 3.0)),
                 include_angular_velocity: bool = False,
                 # ---- logging / visualization ------------------------------
                 log_video: bool = False,
                 video_size=(64, 64),
                 # ---- reward weights ---------------------------------------
                 reward_cfg=None,
                 ):
        # Task configuration ---------------------------------------------------
        self.GOAL_TOLERANCE = float(goal_tolerance)
        self.EPISODE_LEN_SEC = int(episode_len_sec)
        self.RANDOMIZE_GOAL = bool(randomize_goal)
        self.RANDOMIZE_START = bool(randomize_start)
        self.GOAL_RANGE = np.array(goal_sample_range, dtype=np.float32)
        self.START_RANGE = np.array(start_sample_range, dtype=np.float32)
        self.BOUNDS = np.array(bounds, dtype=np.float32)
        self.INCLUDE_ANG_VEL = bool(include_angular_velocity)

        # Visualization (third-person RGB frames for DreamerV3 log/image) -----
        self.LOG_VIDEO = bool(log_video)
        self.VIDEO_SIZE = (int(video_size[0]), int(video_size[1]))  # (H, W)

        self._fixed_goal = None if goal_pos is None else np.array(goal_pos, dtype=np.float32)
        self._fixed_start = None if start_pos is None else np.array(start_pos, dtype=np.float32)

        # Reward weights (override individually via reward_cfg dict) -----------
        self.RW = {
            "progress": 1.0,     # reward per meter of distance reduction toward goal
            "goal_bonus": 10.0,  # one-off reward when goal is reached
            "time_penalty": 0.0,    # constant per-step penalty (encourages speed)
            "crash_penalty": 10.0,  # penalty on termination by crash / out-of-bounds
            "action_smooth": 0.0,   # penalty on change of action between steps
            "tilt_penalty": 0.0,    # penalty proportional to roll/pitch magnitude
            "alive": 0.0,           # constant per-step survival reward
        }
        if reward_cfg:
            self.RW.update(reward_cfg)

        # Goal / start placeholders (filled in reset) -------------------------
        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._prev_dist = None
        self._prev_action = None

        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL)

    ################################################################################
    # Goal / start sampling
    ################################################################################

    def _sample_in_range(self, rng_box):
        return np.array([self.np_random.uniform(lo, hi) for lo, hi in rng_box],
                        dtype=np.float32)

    def _resample_task(self):
        """Pick a new goal (and optionally start) for the next episode."""
        if self._fixed_goal is not None and not self.RANDOMIZE_GOAL:
            self.TARGET_POS = self._fixed_goal.copy()
        else:
            self.TARGET_POS = self._sample_in_range(self.GOAL_RANGE)

        if self.RANDOMIZE_START:
            start = (self._fixed_start.copy() if self._fixed_start is not None
                     else self._sample_in_range(self.START_RANGE))
            self.INIT_XYZS = start.reshape(1, 3)

    ################################################################################

    def reset(self, seed=None, options=None):
        # Sample the new task BEFORE BaseAviary rebuilds the simulation, so the
        # drone spawns at the (possibly randomized) start pose. Accessing
        # `self.np_random` lazily initializes the RNG on the first episode.
        if seed is not None:
            super().reset(seed=seed)
        self._resample_task()
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self._prev_dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._prev_action = None
        return obs, info

    ################################################################################
    # Observation
    ################################################################################

    def _observationSpace(self):
        dim = 12 if self.INCLUDE_ANG_VEL else 9
        hi = np.inf * np.ones(dim, dtype=np.float32)
        return spaces.Box(low=-hi, high=hi, shape=(dim,), dtype=np.float32)

    def _computeObs(self):
        s = self._getDroneStateVector(0)
        rel_goal = self.TARGET_POS - s[0:3]   # (3,)
        vel = s[10:13]                         # (3,)
        rpy = s[7:10]                          # (3,)
        parts = [rel_goal, vel, rpy]
        if self.INCLUDE_ANG_VEL:
            parts.append(s[13:16])             # body angular velocity
        return np.concatenate(parts).astype(np.float32)

    ################################################################################
    # Reward
    ################################################################################

    def _distance_to_goal(self):
        s = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.TARGET_POS - s[0:3]))

    def _computeReward(self):
        s = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - s[0:3]))

        # 1) Potential-based progress shaping: positive when getting closer.
        if self._prev_dist is None:
            self._prev_dist = dist
        progress = self._prev_dist - dist
        reward = self.RW["progress"] * progress
        self._prev_dist = dist

        # 2) Constant terms.
        reward += self.RW["alive"]
        reward -= self.RW["time_penalty"]

        # 3) Tilt penalty (discourage aggressive attitudes).
        if self.RW["tilt_penalty"]:
            roll, pitch = s[7], s[8]
            reward -= self.RW["tilt_penalty"] * (abs(roll) + abs(pitch))

        # 4) Action smoothness penalty.
        if self.RW["action_smooth"] and len(self.action_buffer) >= 2:
            a_now = np.asarray(self.action_buffer[-1][0])
            a_prev = np.asarray(self.action_buffer[-2][0])
            reward -= self.RW["action_smooth"] * float(np.linalg.norm(a_now - a_prev))

        # 5) Goal bonus.
        if dist < self.GOAL_TOLERANCE:
            reward += self.RW["goal_bonus"]

        # 6) Crash / out-of-bounds penalty (mirrors _computeTerminated).
        if self._is_crash(s):
            reward -= self.RW["crash_penalty"]

        return float(reward)

    ################################################################################
    # Termination / truncation
    ################################################################################

    def _is_out_of_bounds(self, s):
        x, y, z = s[0], s[1], s[2]
        (xl, xh), (yl, yh), (zl, zh) = self.BOUNDS
        return (x < xl or x > xh or y < yl or y > yh or z < zl or z > zh)

    def _is_crash(self, s):
        # Excessive tilt (> ~70 deg) or out of the allowed flight volume.
        too_tilted = abs(s[7]) > 1.2 or abs(s[8]) > 1.2
        return self._is_out_of_bounds(s) or too_tilted

    def _computeTerminated(self):
        s = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - s[0:3]) < self.GOAL_TOLERANCE:
            return True   # success
        if self._is_crash(s):
            return True   # failure
        return False

    def _computeTruncated(self):
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        s = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - s[0:3]))
        success = dist < self.GOAL_TOLERANCE
        crash = self._is_crash(s)
        return {
            "distance": dist,
            "is_success": bool(success),
            # FromGym/embodied uses is_terminal to mask bootstrapping; a
            # time-limit truncation is NOT terminal, a crash/success is.
            "is_terminal": bool(success or crash),
            "goal": self.TARGET_POS.copy(),
        }

    ################################################################################
    # Visualization
    ################################################################################

    @property
    def video_shape(self):
        """(H, W, 3) shape of the third-person RGB frames."""
        return (self.VIDEO_SIZE[0], self.VIDEO_SIZE[1], 3)

    def render_frame(self):
        """Render a third-person RGB frame following the drone and goal.

        Returns a ``uint8`` array of shape ``video_shape`` suitable for the
        DreamerV3 ``log/image`` key (logged as a video by the train loop).
        """
        h, w = self.VIDEO_SIZE
        s = self._getDroneStateVector(0)
        drone_pos = s[0:3]
        center = 0.5 * (drone_pos + self.TARGET_POS)
        view = p.computeViewMatrixFromYawPitchRoll(
            distance=2.5, yaw=-30, pitch=-30, roll=0,
            cameraTargetPosition=center, upAxisIndex=2,
            physicsClientId=self.CLIENT)
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=float(w) / float(h), nearVal=0.1, farVal=1000.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width=w, height=h, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.CLIENT)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3]
        return rgb.astype(np.uint8)

