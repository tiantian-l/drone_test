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
import os
import pkgutil
from collections import deque
from sys import platform

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
                 render_mode: str = "3d",
                 video_size=(256, 256),
                 trail_length: int = 80,
                 trail_markers: int = 30,
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
        # render_mode == "3d" -> photorealistic PyBullet camera (GPU OpenGL on
        # a Tesla T4 via the EGL plugin, CPU TinyRenderer otherwise);
        # render_mode == "2d" -> the lightweight top-down numpy schematic.
        self.LOG_VIDEO = bool(log_video)
        self.RENDER_MODE = str(render_mode)
        self.VIDEO_SIZE = (int(video_size[0]), int(video_size[1]))  # (H, W)
        self.TRAIL_LENGTH = int(trail_length)
        self.TRAIL_MARKERS = int(trail_markers)
        self._trail = deque(maxlen=self.TRAIL_LENGTH)
        self._trail_bodies = []      # visual-only spheres marking the path
        self._drone_marker_id = None  # cyan highlight so the tiny drone shows
        self._egl_plugin = None
        self._pyb_renderer = None    # chosen in _setup_offscreen_renderer
        self._cam_yaw = 45.0         # slowly orbits for better depth cues

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

        # Set up the offscreen renderer used for the third-person video. On a
        # Linux GPU box (e.g. Tesla T4) this loads PyBullet's EGL plugin so the
        # frames are rendered on the GPU (ER_BULLET_HARDWARE_OPENGL); otherwise
        # it transparently falls back to the CPU TinyRenderer.
        if self.LOG_VIDEO and self.RENDER_MODE == "3d":
            self._setup_offscreen_renderer()

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
        self._trail.clear()
        self._trail.append(state[0:3].copy())
        # BaseAviary.reset() calls p.resetSimulation(), which wipes every body
        # (including our trail/drone markers); drop the stale ids so they are
        # lazily recreated on the next render.
        self._trail_bodies = []
        self._drone_marker_id = None
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

    def _addObstacles(self):
        """Add a visual-only marker at the goal so it is visible in renders.

        Called by BaseAviary during housekeeping (after the plane and drone are
        loaded). The marker has no mass and no collision shape, so it does not
        affect the physics or the drone state vector. TARGET_POS is set by
        `_resample_task()` before `super().reset()` triggers housekeeping.
        """
        super()._addObstacles()
        try:
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.08, rgbaColor=[1.0, 0.1, 0.1, 1.0],
                physicsClientId=self.CLIENT)
            self._goal_marker_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=self.TARGET_POS.tolist(),
                physicsClientId=self.CLIENT)
        except Exception:
            self._goal_marker_id = None

    @property
    def video_shape(self):
        """(H, W, 3) shape of the third-person RGB frames."""
        return (self.VIDEO_SIZE[0], self.VIDEO_SIZE[1], 3)

    # ---- 3D camera (GPU/CPU PyBullet) ---------------------------------------

    def _setup_offscreen_renderer(self):
        """Pick the offscreen renderer; prefer the GPU on Linux (Tesla T4).

        Tries PyBullet's EGL plugin so ``getCameraImage`` renders on the GPU
        with ``ER_BULLET_HARDWARE_OPENGL``. Headless cloud containers often
        fail to create an EGL context ("failed to EGL with glad"); crucially
        ``loadPlugin`` can still return a valid id in that case and only the
        actual render comes back black. We therefore *verify* EGL with a small
        test render and fall back to the CPU ``ER_TINY_RENDERER`` if it does
        not produce a real image. The choice can be forced via the env var
        ``DRONE_RENDERER`` (``egl`` | ``tiny`` | ``auto``, default ``auto``).
        """
        self._pyb_renderer = p.ER_TINY_RENDERER
        choice = os.environ.get("DRONE_RENDERER", "auto").strip().lower()
        if choice == "tiny" or platform != "linux":
            return
        try:
            egl = pkgutil.get_loader("eglRenderer")
            if egl is None:
                return
            self._egl_plugin = p.loadPlugin(
                egl.get_filename(), "_eglRendererPlugin",
                physicsClientId=self.CLIENT)
            if self._egl_plugin is None or self._egl_plugin < 0:
                self._egl_plugin = None
                return
            if choice == "egl":
                # Trust the user; skip verification.
                self._pyb_renderer = p.ER_BULLET_HARDWARE_OPENGL
                return
            # auto: verify the GPU context actually renders a non-black frame.
            if self._egl_test_render_ok():
                self._pyb_renderer = p.ER_BULLET_HARDWARE_OPENGL
                print("[NavigationAviary] video renderer: GPU EGL "
                      "(ER_BULLET_HARDWARE_OPENGL)")
            else:
                self._unload_egl()
                print("[NavigationAviary] EGL context unusable; falling back "
                      "to CPU TinyRenderer for video frames.")
        except Exception:
            self._unload_egl()
            self._pyb_renderer = p.ER_TINY_RENDERER

    def _egl_test_render_ok(self):
        """Render a tiny probe frame; True iff EGL returns a real image."""
        try:
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5], distance=2.0,
                yaw=45.0, pitch=-30.0, roll=0.0, upAxisIndex=2,
                physicsClientId=self.CLIENT)
            proj = p.computeProjectionMatrixFOV(
                fov=60.0, aspect=1.0, nearVal=0.05, farVal=100.0,
                physicsClientId=self.CLIENT)
            _, _, rgb, _, _ = p.getCameraImage(
                width=32, height=32, viewMatrix=view, projectionMatrix=proj,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                flags=p.ER_NO_SEGMENTATION_MASK,
                physicsClientId=self.CLIENT)
            arr = np.reshape(np.asarray(rgb, dtype=np.uint8), (32, 32, 4))[:, :, :3]
            # A working render of the ground + drone has spatial variation; a
            # failed EGL context returns a uniform (usually black) buffer.
            return bool(arr.max() > 0 and arr.std() > 1.0)
        except Exception:
            return False

    def _unload_egl(self):
        try:
            if self._egl_plugin is not None and self._egl_plugin >= 0:
                p.unloadPlugin(self._egl_plugin, physicsClientId=self.CLIENT)
        except Exception:
            pass
        self._egl_plugin = None
        self._pyb_renderer = p.ER_TINY_RENDERER


    def _update_trail_markers(self):
        """Place small visual-only spheres along the recent flight path."""
        try:
            n = self.TRAIL_MARKERS
            if n <= 0:
                return
            if not self._trail_bodies:
                for _ in range(n):
                    vis = p.createVisualShape(
                        p.GEOM_SPHERE, radius=0.022,
                        rgbaColor=[0.20, 0.55, 1.0, 0.9],
                        physicsClientId=self.CLIENT)
                    bid = p.createMultiBody(
                        baseMass=0, baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=vis,
                        basePosition=[0.0, 0.0, -10.0],
                        physicsClientId=self.CLIENT)
                    self._trail_bodies.append(bid)
            # Evenly subsample the trail down to the marker pool size.
            pts = list(self._trail)
            if len(pts) > n:
                idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
                pts = [pts[i] for i in idx]
            for i, bid in enumerate(self._trail_bodies):
                pos = pts[i].tolist() if i < len(pts) else [0.0, 0.0, -10.0]
                p.resetBasePositionAndOrientation(
                    bid, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
        except Exception:
            pass

    def _update_drone_marker(self, drone_pos):
        """Place a cyan highlight sphere on the drone so it is easy to spot."""
        try:
            if self._drone_marker_id is None:
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.06,
                    rgbaColor=[0.10, 0.85, 0.95, 0.85],
                    physicsClientId=self.CLIENT)
                self._drone_marker_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=drone_pos.tolist(),
                    physicsClientId=self.CLIENT)
            else:
                p.resetBasePositionAndOrientation(
                    self._drone_marker_id, drone_pos.tolist(), [0, 0, 0, 1],
                    physicsClientId=self.CLIENT)
        except Exception:
            pass

    def _render_camera_3d(self):
        """Render a tracking 3D perspective view of the drone and goal."""
        h, w = self.VIDEO_SIZE
        s = self._getDroneStateVector(0)
        drone_pos = s[0:3]
        goal = self.TARGET_POS
        if len(self._trail) == 0 or np.linalg.norm(self._trail[-1] - drone_pos) > 1e-3:
            self._trail.append(drone_pos.copy())
        self._update_trail_markers()
        self._update_drone_marker(drone_pos)

        # Frame both the drone and the goal: look at their midpoint and back
        # the camera off proportionally to their separation.
        target = (0.5 * (drone_pos + goal)).tolist()
        sep = float(np.linalg.norm(drone_pos - goal))
        distance = float(np.clip(1.6 + 0.8 * sep, 2.0, 7.0))
        self._cam_yaw = (self._cam_yaw + 0.35) % 360.0

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=distance,
            yaw=self._cam_yaw, pitch=-32.0, roll=0.0, upAxisIndex=2,
            physicsClientId=self.CLIENT)
        proj = p.computeProjectionMatrixFOV(
            fov=60.0, aspect=float(w) / float(h), nearVal=0.05, farVal=100.0,
            physicsClientId=self.CLIENT)
        _, _, rgb, _, _ = p.getCameraImage(
            width=w, height=h, viewMatrix=view, projectionMatrix=proj,
            renderer=self._pyb_renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.CLIENT)
        rgb = np.reshape(np.asarray(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]
        return np.ascontiguousarray(rgb)

    # ---- 2D schematic (numpy, renderer-independent) -------------------------

    def _world_to_pixel(self, x, y):
        (xl, xh), (yl, yh), _ = self.BOUNDS
        h, w = self.VIDEO_SIZE
        pad = 0.08
        span_x = max(xh - xl, 1e-6)
        span_y = max(yh - yl, 1e-6)
        nx = (x - xl) / span_x
        ny = (y - yl) / span_y
        px = int(np.clip((pad + (1.0 - 2.0 * pad) * nx) * (w - 1), 0, w - 1))
        py = int(np.clip((pad + (1.0 - 2.0 * pad) * (1.0 - ny)) * (h - 1), 0, h - 1))
        return px, py

    def _draw_disc(self, image, cx, cy, radius, color):
        h, w, _ = image.shape
        x0 = max(0, cx - radius)
        x1 = min(w - 1, cx + radius)
        y0 = max(0, cy - radius)
        y1 = min(h - 1, cy + radius)
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        image[y0:y1 + 1, x0:x1 + 1][mask] = color

    def _draw_line(self, image, p0, p1, color, thickness=1):
        x0, y0 = p0
        x1, y1 = p1
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, steps).round().astype(int)
        ys = np.linspace(y0, y1, steps).round().astype(int)
        for x, y in zip(xs, ys):
            self._draw_disc(image, int(x), int(y), thickness, color)

    def _draw_grid(self, image):
        (xl, xh), (yl, yh), _ = self.BOUNDS
        step = 1.0
        color = np.array([36, 40, 48], dtype=np.uint8)
        x = np.ceil(xl / step) * step
        while x <= xh + 1e-6:
            p0 = self._world_to_pixel(x, yl)
            p1 = self._world_to_pixel(x, yh)
            self._draw_line(image, (p0[0], p0[1]), (p1[0], p1[1]), color, thickness=0)
            x += step
        y = np.ceil(yl / step) * step
        while y <= yh + 1e-6:
            p0 = self._world_to_pixel(xl, y)
            p1 = self._world_to_pixel(xh, y)
            self._draw_line(image, (p0[0], p0[1]), (p1[0], p1[1]), color, thickness=0)
            y += step

    def render_frame(self):
        """Return one third-person RGB frame for the DreamerV3 video log.

        In ``3d`` mode this is a tracking PyBullet camera (GPU on a Tesla T4,
        CPU otherwise); any failure degrades gracefully to the 2D schematic.
        """
        if self.RENDER_MODE == "3d":
            try:
                return self._render_camera_3d()
            except Exception:
                pass  # fall back to the renderer-independent schematic
        return self._render_schematic()

    def _render_schematic(self):
        """Render a clear top-down schematic from state only.

        This avoids the tiny/blurred PyBullet camera view and keeps the video
        readable even on fast training runs.
        """
        h, w = self.VIDEO_SIZE
        image = np.full((h, w, 3), 244, dtype=np.uint8)
        self._draw_grid(image)

        border = np.array([160, 168, 180], dtype=np.uint8)
        image[0:2, :, :] = border
        image[-2:, :, :] = border
        image[:, 0:2, :] = border
        image[:, -2:, :] = border

        if len(self._trail) >= 2:
            trail_color = np.array([52, 120, 220], dtype=np.uint8)
            trail = list(self._trail)
            for p0, p1 in zip(trail[:-1], trail[1:]):
                x0, y0 = self._world_to_pixel(float(p0[0]), float(p0[1]))
                x1, y1 = self._world_to_pixel(float(p1[0]), float(p1[1]))
                self._draw_line(image, (x0, y0), (x1, y1), trail_color, thickness=1)

        s = self._getDroneStateVector(0)
        drone_pos = s[0:3]
        goal = self.TARGET_POS
        if len(self._trail) == 0 or np.linalg.norm(self._trail[-1] - drone_pos) > 1e-4:
            self._trail.append(drone_pos.copy())

        gx, gy = self._world_to_pixel(float(goal[0]), float(goal[1]))
        self._draw_disc(image, gx, gy, radius=4, color=np.array([230, 55, 55], dtype=np.uint8))
        self._draw_disc(image, gx, gy, radius=2, color=np.array([255, 220, 220], dtype=np.uint8))

        dx, dy = self._world_to_pixel(float(drone_pos[0]), float(drone_pos[1]))
        self._draw_disc(image, dx, dy, radius=5, color=np.array([35, 200, 210], dtype=np.uint8))
        self._draw_disc(image, dx, dy, radius=2, color=np.array([240, 250, 255], dtype=np.uint8))

        yaw = float(s[9])
        arrow_len = 14.0
        hx = dx + int(round(arrow_len * np.cos(yaw)))
        hy = dy - int(round(arrow_len * np.sin(yaw)))
        self._draw_line(image, (dx, dy), (hx, hy), np.array([20, 70, 90], dtype=np.uint8), thickness=1)

        z_norm = float(np.clip((drone_pos[2] - self.BOUNDS[2, 0]) / max(self.BOUNDS[2, 1] - self.BOUNDS[2, 0], 1e-6), 0.0, 1.0))
        bar_h = int(round(20 + 40 * z_norm))
        image[h - 8 - bar_h:h - 8, w - 8:w - 4, :] = np.array([60, 180, 90], dtype=np.uint8)

        return image

