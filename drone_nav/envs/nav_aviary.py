"""A->B point navigation environment for a single quadrotor.

Built on top of gym-pybullet-drones `BaseRLAviary` using the *velocity*
controller (`ActionType.VEL`).  The observation is purely proprioceptive
(state-only), as requested:

    obs = [
        rel_x, rel_y, rel_z,     # goal_pos - drone_pos      (3)
        vx, vy, vz,              # linear velocity (world)   (3)
        roll, pitch, yaw,        # attitude (euler)          (3)
        wx, wy, wz,              # body angular velocity      (3, optional)
        lidar[0..N-1],           # 3D lidar ranges (N=hbeams*vbeams, optional)
    ]

The lidar is a 3D multi-layer scan: ``lidar_vbeams`` elevation layers spanning
``lidar_vfov`` (deg), each with ``360 / lidar_hres_deg`` horizontal beams.

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
                 randomize_start: bool = True,
                 # 20m x 20m arena (map half-range = [10, 10, 4.5]). Obstacles
                 # fill the +/-10 field; the start is sampled ~2m beyond the
                 # field on the LEFT (x ~ -12), the goal ~2m beyond on the
                 # RIGHT (x ~ +12), both randomized along y so every episode is
                 # a different ~24m crossing of the obstacle forest.
                 goal_sample_range=((11.7, 12.0), (-9.0, 9.0), (0.6, 3.5)),
                 start_sample_range=((-12.0, -11.7), (-9.0, 9.0), (0.8, 1.5)),
                 goal_tolerance: float = 0.5,
                 # A ~24 m crossing at a 2.0 m/s velocity cap needs ~12 s in a
                 # straight line; 1200 control steps (= 40 s at ctrl_freq 30)
                 # leaves ample margin to weave around the obstacles.
                 episode_len_sec: int = 40,
                 # Flight limits (out-of-bounds crash volume). Slightly larger
                 # than the +/-12 start/goal band on x so spawning at the edge
                 # is not an instant crash; z ceiling = map height (4.5 m).
                 bounds=((-12.5, 12.5), (-10.5, 10.5), (0.05, 4.5)),
                 # Obstacle placement field half-range [x, y, z_ceiling].
                 map_range=(10.0, 10.0, 4.5),
                 include_angular_velocity: bool = False,
                 # Optional override (m/s) of the DSL-PID velocity controller's
                 # speed cap. ``None`` keeps gym-pybullet-drones' default
                 # (0.25 m/s for CF2X); the 2.0 m/s default here lets the drone
                 # cross the larger arena in a reasonable horizon.
                 speed_limit=2.0,
                 # ---- deterministic evaluation -----------------------------
                 # When True the whole task (start / goal / obstacles) is
                 # regenerated from a FIXED per-env seed on every reset, so the
                 # evaluation set is identical across eval cycles. Training envs
                 # keep eval_mode=False and re-randomize every episode.
                 eval_mode: bool = False,
                 eval_seed: int = 0,
                 # ---- obstacles (static collision boxes) -------------------
                 obstacles_enabled: bool = True,
                 n_obstacles: int = 88,
                 # Box footprint: width in x and y sampled independently.
                 obstacle_width_range=(0.4, 1.1),
                 # Height is sampled from a mixture so most pillars span the
                 # full 4.5m ceiling (must be flown around) while a minority are
                 # short enough to fly over. Each entry is (prob, (lo, hi)).
                 obstacle_height_dist=((0.10, (1.0, 1.5)),
                                       (0.15, (1.5, 2.0)),
                                       (0.20, (2.0, 4.0)),
                                       (0.55, (4.0, 6.0))),
                 obstacle_clear_radius: float = 0.8,
                 obstacle_min_separation: float = 0.4,
                 collision_distance: float = 0.3,
                 # ---- guaranteed-feasible map (grid BFS) -------------------
                 # After sampling, a start->goal path is verified on an inflated
                 # occupancy grid; layouts without a corridor are resampled so
                 # the goal is always reachable.
                 path_clearance: float = 0.35,
                 path_cell_size: float = 0.25,
                 obstacle_layout_tries: int = 40,
                 # ---- lidar (3D multi-layer range scan) --------------------
                 lidar_enabled: bool = True,
                 lidar_range: float = 4.0,
                 lidar_hres_deg: float = 10.0,
                 lidar_vfov=(-10.0, 20.0),
                 lidar_vbeams: int = 4,
                 lidar_start_offset: float = 0.06,
                 # ---- logging / visualization ------------------------------
                 log_video: bool = False,
                 render_mode: str = "3d",
                 video_size=(512, 512),
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
        self.MAP_RANGE = np.array(map_range, dtype=np.float32)
        self.INCLUDE_ANG_VEL = bool(include_angular_velocity)
        self.SPEED_LIMIT_OVERRIDE = None if speed_limit is None else float(speed_limit)

        # Deterministic-eval configuration ------------------------------------
        self.EVAL_MODE = bool(eval_mode)
        self.EVAL_SEED = int(eval_seed)

        # Obstacle configuration ----------------------------------------------
        self.OBSTACLES_ENABLED = bool(obstacles_enabled)
        self.N_OBSTACLES = int(n_obstacles)
        self.OBSTACLE_WIDTH_RANGE = (float(obstacle_width_range[0]),
                                     float(obstacle_width_range[1]))
        # Normalize the height mixture to (cumulative_prob, (lo, hi)) buckets.
        self.OBSTACLE_HEIGHT_DIST = tuple(
            (float(p), (float(lo), float(hi))) for p, (lo, hi) in obstacle_height_dist)
        self.OBSTACLE_CLEAR_RADIUS = float(obstacle_clear_radius)
        self.OBSTACLE_MIN_SEPARATION = float(obstacle_min_separation)
        self.COLLISION_DISTANCE = float(collision_distance)
        # Feasibility-check knobs (grid BFS over an inflated occupancy map).
        self.PATH_CLEARANCE = float(path_clearance)
        self.PATH_CELL_SIZE = float(path_cell_size)
        self.OBSTACLE_LAYOUT_TRIES = int(obstacle_layout_tries)
        self._obstacle_specs = []     # (x, y, wx, wy, h) boxes for this episode
        self._obstacle_ids = []       # PyBullet body ids (rebuilt each reset)
        self._clearance_cache = (-1, np.inf)  # (step_counter, min surface dist)
        self._last_lidar = None       # most recent normalized scan (cached)
        self._min_lidar = np.inf      # closest lidar reading (m) this episode

        # Lidar configuration (3D multi-layer scan) ---------------------------
        self.LIDAR_ENABLED = bool(lidar_enabled)
        self.LIDAR_RANGE = float(lidar_range)
        self.LIDAR_HRES_DEG = float(lidar_hres_deg)
        self.LIDAR_VFOV = (float(lidar_vfov[0]), float(lidar_vfov[1]))
        self.LIDAR_VBEAMS = int(lidar_vbeams)
        self.LIDAR_START_OFFSET = float(lidar_start_offset)
        # Horizontal beams evenly spaced over the full circle at LIDAR_HRES_DEG.
        self.LIDAR_H_BEAMS = max(1, int(round(360.0 / self.LIDAR_HRES_DEG)))
        self.LIDAR_N_BEAMS = self.LIDAR_H_BEAMS * self.LIDAR_VBEAMS
        self._lidar_h_angles = np.deg2rad(
            np.arange(self.LIDAR_H_BEAMS, dtype=np.float32) * self.LIDAR_HRES_DEG)
        # Elevation angles across the vertical FOV (inclusive endpoints).
        if self.LIDAR_VBEAMS > 1:
            self._lidar_v_angles = np.deg2rad(np.linspace(
                self.LIDAR_VFOV[0], self.LIDAR_VFOV[1],
                self.LIDAR_VBEAMS).astype(np.float32))
        else:
            self._lidar_v_angles = np.deg2rad(
                np.array([0.5 * (self.LIDAR_VFOV[0] + self.LIDAR_VFOV[1])],
                         dtype=np.float32))

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
        self._drone_marker_id = None  # subtle halo so the tiny drone shows
        self._heading_marker_id = None  # visual-only yaw direction shaft
        self._heading_tip_id = None     # visual-only yaw direction tip
        self._egl_plugin = None
        self._pyb_renderer = None    # chosen in _setup_offscreen_renderer
        self._cam_yaw = 45.0         # slowly orbits for better depth cues

        self._fixed_goal = None if goal_pos is None else np.array(goal_pos, dtype=np.float32)
        self._fixed_start = None if start_pos is None else np.array(start_pos, dtype=np.float32)

        # Reward weights (override individually via reward_cfg dict) -----------
        self.RW = {
            #"progress": 1.0,     # reward per meter of distance reduction toward goal
            # "progress": 5.0,
            # "goal_bonus": 10.0,  # one-off reward when goal is reached
           # "time_penalty": 0.0,    # constant per-step penalty (encourages speed)
            # "time_penalty": 0.01, 
            # "crash_penalty": 10.0,  # penalty on termination by crash / out-of-bounds
            # "collision_penalty": 10.0,  # one-off penalty when hitting an obstacle
            # "obstacle_proximity": 0.5,  # per-step penalty inside the safety margin
            # "safety_margin": 0.30,      # distance (m) where proximity penalty starts
            # "action_smooth": 0.0,   # penalty on change of action between steps
            # "tilt_penalty": 0.0,    # penalty proportional to roll/pitch magnitude
            # "alive": 0.0,           # constant per-step survival reward
            
            "progress": 8.0,
            "goal_bonus": 30.0,
            "time_penalty": 0.02,
            "timeout_penalty": 10.0,
            "crash_penalty": 10.0,
            "collision_penalty": 10.0,
            "obstacle_proximity": 0.2,
            "safety_margin": 0.5,
            "action_smooth": 0.0,
            "tilt_penalty": 0.0,
            "alive": 0.0,
        }
        
   
        if reward_cfg:
            self.RW.update(reward_cfg)

        # Goal / start placeholders (filled in reset) -------------------------
        self.TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._prev_dist = None
        self._prev_action = None
        self._min_dist = None         # closest the drone got to the goal so far
        self._reward_terms = {}       # per-step reward breakdown (for logging)

        if initial_xyzs is None:
            initial_xyzs = np.array([[-12.0, 0.0, 1.0]], dtype=np.float32)

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

        # Optionally raise the DSL-PID velocity cap so the drone can cross the
        # larger arena faster (BaseRLAviary sets SPEED_LIMIT for ActionType.VEL
        # to ~0.25 m/s for CF2X, which is very slow over 10 m).
        if self.SPEED_LIMIT_OVERRIDE is not None:
            self.SPEED_LIMIT = self.SPEED_LIMIT_OVERRIDE

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

        # Sample the obstacle layout for the upcoming episode. Only the
        # positions are chosen here; the PyBullet bodies are created later in
        # `_addObstacles` (after `super().reset()` rebuilds the simulation).
        self._sample_obstacles()

    ################################################################################

    def _sample_obstacles(self):
        """Sample static box obstacles for the upcoming episode.

        Obstacles fill the +/-``MAP_RANGE`` field between the left-side start
        and the right-side goal so the drone must weave through them. Start and
        goal neighbourhoods are kept clear and a minimum separation preserves
        gaps between pillars. Crucially, every candidate layout is validated
        with a grid BFS (``_path_exists``): if the pillars would wall the goal
        off, the layout is resampled until a traversable corridor remains, so
        the goal is always reachable.
        """
        self._obstacle_specs = []
        if not self.OBSTACLES_ENABLED or self.N_OBSTACLES <= 0:
            return

        start_xy = np.asarray(self.INIT_XYZS[0][:2], dtype=np.float32)
        goal_xy = np.asarray(self.TARGET_POS[:2], dtype=np.float32)

        # Try whole layouts until one is provably feasible.
        best = []
        for _ in range(self.OBSTACLE_LAYOUT_TRIES):
            specs = self._try_sample_layout(start_xy, goal_xy)
            if self._path_exists(start_xy, goal_xy, specs):
                self._obstacle_specs = specs
                return
            best = specs
        # Fallback: drop pillars from the last (infeasible) layout one by one
        # until a corridor opens up, guaranteeing the goal stays reachable.
        while best and not self._path_exists(start_xy, goal_xy, best):
            best.pop()
        self._obstacle_specs = best

    def _sample_obstacle_height(self):
        """Sample a pillar height from the configured mixture distribution."""
        u = float(self.np_random.uniform(0.0, 1.0))
        acc = 0.0
        for prob, (lo, hi) in self.OBSTACLE_HEIGHT_DIST:
            acc += prob
            if u <= acc:
                return float(self.np_random.uniform(lo, hi))
        # Numerical fallback (probs not summing to exactly 1.0).
        lo, hi = self.OBSTACLE_HEIGHT_DIST[-1][1]
        return float(self.np_random.uniform(lo, hi))

    def _try_sample_layout(self, start_xy, goal_xy):
        """Sample one candidate box-obstacle layout (no feasibility guarantee).

        Each spec is ``(x, y, wx, wy, h)``: centre, x/y footprint widths and
        height. Obstacles are confined to the +/-``MAP_RANGE`` field.
        """
        mx, my = float(self.MAP_RANGE[0]), float(self.MAP_RANGE[1])
        wlo, whi = self.OBSTACLE_WIDTH_RANGE
        specs = []
        # Keep box centres inside the field, leaving room for the widest box.
        wall = 0.5 * whi
        x_lo, x_hi = -mx + wall, mx - wall
        y_lo, y_hi = -my + wall, my - wall

        def _bound_radius(wx, wy):
            return 0.5 * float(np.hypot(wx, wy))

        def _ok(x, y, wx, wy):
            q = np.array([x, y], dtype=np.float32)
            r = _bound_radius(wx, wy)
            if np.linalg.norm(q - start_xy) < self.OBSTACLE_CLEAR_RADIUS + r:
                return False
            if np.linalg.norm(q - goal_xy) < self.OBSTACLE_CLEAR_RADIUS + r:
                return False
            for (ox, oy, owx, owy, _oh) in specs:
                if np.linalg.norm(q - np.array([ox, oy], dtype=np.float32)) < (
                        self.OBSTACLE_MIN_SEPARATION + r + _bound_radius(owx, owy)):
                    return False
            return True

        max_tries = 200
        while len(specs) < self.N_OBSTACLES:
            placed = False
            for _try in range(max_tries):
                x = float(self.np_random.uniform(x_lo, x_hi))
                y = float(self.np_random.uniform(y_lo, y_hi))
                wx = float(self.np_random.uniform(wlo, whi))
                wy = float(self.np_random.uniform(wlo, whi))
                if _ok(x, y, wx, wy):
                    specs.append((x, y, wx, wy, self._sample_obstacle_height()))
                    placed = True
                    break
            if not placed:
                break  # arena too crowded; stop trying
        return specs

    def _path_exists(self, start_xy, goal_xy, specs):
        """Return True iff a start->goal path exists around ``specs``.

        Builds an occupancy grid over the flight bounds, marks cells within
        ``half_width + PATH_CLEARANCE`` of any box footprint as blocked (the
        clearance accounts for the drone's own footprint plus a safety margin),
        and runs an 8-connected BFS. An empty layout is trivially feasible.
        """
        if not specs:
            return True
        (xl, xh), (yl, yh), _ = self.BOUNDS
        cell = self.PATH_CELL_SIZE
        nx = max(1, int(np.ceil((xh - xl) / cell)))
        ny = max(1, int(np.ceil((yh - yl) / cell)))
        xs = xl + (np.arange(nx) + 0.5) * cell
        ys = yl + (np.arange(ny) + 0.5) * cell
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        blocked = np.zeros((nx, ny), dtype=bool)
        clr = self.PATH_CLEARANCE
        for (ox, oy, wx, wy, _h) in specs:
            hx = 0.5 * wx + clr
            hy = 0.5 * wy + clr
            blocked |= (np.abs(XX - ox) <= hx) & (np.abs(YY - oy) <= hy)

        def _to_cell(pt):
            i = int(np.clip((pt[0] - xl) / cell, 0, nx - 1))
            j = int(np.clip((pt[1] - yl) / cell, 0, ny - 1))
            return i, j

        si, sj = _to_cell(start_xy)
        gi, gj = _to_cell(goal_xy)
        if blocked[si, sj] or blocked[gi, gj]:
            return False
        visited = np.zeros((nx, ny), dtype=bool)
        visited[si, sj] = True
        queue = deque([(si, sj)])
        neigh = ((-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1))
        while queue:
            i, j = queue.popleft()
            if i == gi and j == gj:
                return True
            for di, dj in neigh:
                ni, nj = i + di, j + dj
                if (0 <= ni < nx and 0 <= nj < ny
                        and not visited[ni, nj] and not blocked[ni, nj]):
                    visited[ni, nj] = True
                    queue.append((ni, nj))
        return False

    ################################################################################

    def reset(self, seed=None, options=None):
        # Sample the new task BEFORE BaseAviary rebuilds the simulation, so the
        # drone spawns at the (possibly randomized) start pose. Accessing
        # `self.np_random` lazily initializes the RNG on the first episode.
        #
        # In eval mode we FORCE the fixed per-env seed on every reset, so the
        # RNG is re-seeded identically each episode and the whole task
        # (start / goal / obstacles) is reproduced exactly -> a stable, directly
        # comparable evaluation set across eval cycles.
        if self.EVAL_MODE:
            seed = self.EVAL_SEED
        if seed is not None:
            super().reset(seed=seed)
        self._resample_task()
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        d0 = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        self._prev_dist = d0
        self._min_dist = d0
        self._prev_action = None
        self._reward_terms = {}
        self._trail.clear()
        self._trail.append(state[0:3].copy())
        # BaseAviary.reset() calls p.resetSimulation(), which wipes every body
        # (including our trail/drone markers); drop the stale ids so they are
        # lazily recreated on the next render.
        self._trail_bodies = []
        self._drone_marker_id = None
        self._heading_marker_id = None
        self._heading_tip_id = None
        # `_addObstacles` (run inside super().reset() housekeeping) already
        # rebuilt `self._obstacle_ids`; only the per-step cache needs clearing.
        self._clearance_cache = (-1, np.inf)
        self._last_lidar = None
        self._min_lidar = np.inf
        return obs, info

    ################################################################################
    # Observation
    ################################################################################

    def _observationSpace(self):
        base = 12 if self.INCLUDE_ANG_VEL else 9
        low = [-np.inf * np.ones(base, dtype=np.float32)]
        high = [np.inf * np.ones(base, dtype=np.float32)]
        if self.LIDAR_ENABLED:
            # Normalized ranges in [0, 1] (1 == free out to LIDAR_RANGE).
            low.append(np.zeros(self.LIDAR_N_BEAMS, dtype=np.float32))
            high.append(np.ones(self.LIDAR_N_BEAMS, dtype=np.float32))
        low = np.concatenate(low)
        high = np.concatenate(high)
        return spaces.Box(low=low, high=high, shape=low.shape, dtype=np.float32)

    def _computeObs(self):
        s = self._getDroneStateVector(0)
        rel_goal = self.TARGET_POS - s[0:3]   # (3,)
        vel = s[10:13]                         # (3,)
        rpy = s[7:10]                          # (3,)
        parts = [rel_goal, vel, rpy]
        if self.INCLUDE_ANG_VEL:
            parts.append(s[13:16])             # body angular velocity
        if self.LIDAR_ENABLED:
            parts.append(self._read_lidar(s))  # 3D lidar normalized ranges
        return np.concatenate(parts).astype(np.float32)

    def _read_lidar(self, s=None):
        """Cast the 3D lidar (``LIDAR_N_BEAMS`` rays); return normalized ranges.

        The scan has ``LIDAR_VBEAMS`` elevation layers (spanning ``LIDAR_VFOV``)
        each with ``LIDAR_H_BEAMS`` horizontal beams. The returned array is
        flattened as ``[layer0_h0, layer0_h1, ..., layer1_h0, ...]`` and lies in
        [0, 1] where 1.0 means "free out to LIDAR_RANGE" and smaller values mean
        a closer obstacle along that beam. Beams are body-frame (rotated by yaw)
        and ignore the drone itself and the ground plane. With no obstacles the
        scan is all ones.
        """
        n = self.LIDAR_N_BEAMS
        if not self.OBSTACLES_ENABLED or not self._obstacle_ids:
            return np.ones(n, dtype=np.float32)
        if s is None:
            s = self._getDroneStateVector(0)
        pos = s[0:3]
        yaw = float(s[9])
        # Build all (elevation x azimuth) unit directions in the world frame.
        az = self._lidar_h_angles[None, :] + yaw          # (V, H) via broadcast
        el = self._lidar_v_angles[:, None]                # (V, 1)
        cos_el = np.cos(el)
        dx = (cos_el * np.cos(az)).reshape(-1)            # (V*H,)
        dy = (cos_el * np.sin(az)).reshape(-1)
        dz = np.broadcast_to(np.sin(el), (self.LIDAR_VBEAMS,
                                          self.LIDAR_H_BEAMS)).reshape(-1)
        froms = np.stack([pos[0] + self.LIDAR_START_OFFSET * dx,
                          pos[1] + self.LIDAR_START_OFFSET * dy,
                          pos[2] + self.LIDAR_START_OFFSET * dz], axis=1)
        tos = np.stack([pos[0] + self.LIDAR_RANGE * dx,
                        pos[1] + self.LIDAR_RANGE * dy,
                        pos[2] + self.LIDAR_RANGE * dz], axis=1)
        ranges = np.ones(n, dtype=np.float32)
        try:
            results = p.rayTestBatch(
                froms.tolist(), tos.tolist(), physicsClientId=self.CLIENT)
            drone_id = int(self.DRONE_IDS[0])
            for i, r in enumerate(results):
                hit_id = r[0]
                if hit_id in (-1, drone_id, self.PLANE_ID):
                    continue
                ranges[i] = float(r[2])  # hitFraction in [0, 1]
        except Exception:
            pass
        self._last_lidar = ranges
        return ranges

    ################################################################################
    # Reward
    ################################################################################

    def _distance_to_goal(self):
        s = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.TARGET_POS - s[0:3]))

    def _computeReward(self):
        s = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - s[0:3]))

        # Per-step reward breakdown: each entry is the signed contribution of
        # that term to this step's reward. Cached so `_computeInfo` can log it.
        terms = {
            "progress": 0.0,
            "alive": 0.0,
            "time_penalty": 0.0,
            "tilt_penalty": 0.0,
            "action_smooth": 0.0,
            "goal_bonus": 0.0,
            "crash_penalty": 0.0,
            "obstacle_proximity": 0.0,
            "collision_penalty": 0.0,
            "timeout_penalty": 0.0,
        }

        # 1) Potential-based progress shaping: positive when getting closer.
        if self._prev_dist is None:
            self._prev_dist = dist
        progress = self._prev_dist - dist
        terms["progress"] = self.RW["progress"] * progress
        self._prev_dist = dist
        

        # 2) Constant terms.
        terms["alive"] = self.RW["alive"]
        terms["time_penalty"] = -self.RW["time_penalty"]

        # 3) Tilt penalty (discourage aggressive attitudes).
        if self.RW["tilt_penalty"]:
            roll, pitch = s[7], s[8]
            terms["tilt_penalty"] = -self.RW["tilt_penalty"] * (abs(roll) + abs(pitch))

        # 4) Action smoothness penalty.
        if self.RW["action_smooth"] and len(self.action_buffer) >= 2:
            a_now = np.asarray(self.action_buffer[-1][0])
            a_prev = np.asarray(self.action_buffer[-2][0])
            terms["action_smooth"] = -self.RW["action_smooth"] * float(
                np.linalg.norm(a_now - a_prev))

        # 5) Goal bonus.
        if dist < self.GOAL_TOLERANCE:
            terms["goal_bonus"] = self.RW["goal_bonus"]

        # 6) Crash / out-of-bounds penalty (mirrors _computeTerminated).
        if self._is_crash(s):
            terms["crash_penalty"] = -self.RW["crash_penalty"]

        # 7) Obstacle proximity shaping: penalize entering the safety margin so
        #    the agent keeps clearance instead of grazing pillars.
        margin = self.RW["safety_margin"]
        if self.RW["obstacle_proximity"] and margin > 0:
            clearance = self._obstacle_clearance()
            if np.isfinite(clearance) and clearance < margin:
                frac = 1.0 - max(clearance, 0.0) / margin
                terms["obstacle_proximity"] = (
                    -self.RW["obstacle_proximity"] * float(frac))

        # 8) Obstacle collision penalty (mirrors _computeTerminated).
        if self._is_collision():
            terms["collision_penalty"] = -self.RW["collision_penalty"]
            
        # 9) Timeout penalty (mirrors _computeTruncated).
        if self._is_timeout() and not (
            dist < self.GOAL_TOLERANCE or self._is_crash(s) or self._is_collision()
        ):
            terms["timeout_penalty"] = -self.RW["timeout_penalty"]

        self._reward_terms = terms
        return float(sum(terms.values()))

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

    def _obstacle_clearance(self):
        """Min surface distance from the drone to any obstacle (cached per step)."""
        if not self.OBSTACLES_ENABLED or not self._obstacle_ids:
            return np.inf
        if self._clearance_cache[0] == self.step_counter:
            return self._clearance_cache[1]
        min_d = np.inf
        try:
            drone_id = int(self.DRONE_IDS[0])
            query = max(self.COLLISION_DISTANCE, self.RW["safety_margin"]) + 0.05
            for bid in self._obstacle_ids:
                for pt in p.getClosestPoints(
                        drone_id, bid, distance=query,
                        physicsClientId=self.CLIENT):
                    if pt[8] < min_d:
                        min_d = pt[8]
        except Exception:
            pass
        self._clearance_cache = (self.step_counter, min_d)
        return min_d

    def _is_collision(self):
        return self._obstacle_clearance() < self.COLLISION_DISTANCE

    def _computeTerminated(self):
        s = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - s[0:3]) < self.GOAL_TOLERANCE:
            return True   # success
        if self._is_crash(s):
            return True   # failure (out-of-bounds / excessive tilt)
        if self._is_collision():
            return True   # failure (hit an obstacle)
        return False

    def _is_timeout(self):
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeTruncated(self):
        if self._is_timeout():
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        s = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - s[0:3]))
        # Track the closest approach over the whole episode.
        if self._min_dist is None or dist < self._min_dist:
            self._min_dist = dist
        success = dist < self.GOAL_TOLERANCE
        crash = self._is_crash(s)
        collision = self._is_collision()
        # Closest obstacle distance (m) as sensed by the lidar this step, and
        # its running minimum over the episode. With no obstacles in view the
        # scan is all ones -> LIDAR_RANGE. Tracking the episode MIN shows
        # whether the agent learns to keep more clearance over training.
        if self._last_lidar is not None:
            lidar_dist = float(np.min(self._last_lidar)) * self.LIDAR_RANGE
        else:
            lidar_dist = self.LIDAR_RANGE
        if lidar_dist < self._min_lidar:
            self._min_lidar = lidar_dist
        # Timeout is only a "result" when the episode ends by the time limit
        # without first succeeding, crashing, or colliding.
        timeout = bool(self._is_timeout() and not (success or crash or collision))
        return {
            # ---- task result (nav + avoidance metrics) ------------------
            "is_success": bool(success),     # -> log/last/success
            "is_crash": bool(crash),         # -> log/last/crash
            "is_collision": bool(collision), # -> log/last/collision
            "is_timeout": timeout,           # -> log/last/timeout
            "final_distance": dist,          # -> log/last/final_distance
            "min_distance": float(self._min_dist),  # -> log/min/min_distance
            "min_lidar_dist": float(self._min_lidar),  # -> log/min/min_lidar_dist
            # FromGym/embodied uses is_terminal to mask bootstrapping; a
            # time-limit truncation is NOT terminal, a crash/collision/success is.
            "is_terminal": bool(success or crash or collision),
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

        # Static obstacle pillars (real collision bodies, mass 0). Recreated
        # every reset because BaseAviary.reset() wipes the simulation.
        self._obstacle_ids = []
        for (ox, oy, wx, wy, h) in self._obstacle_specs:
            try:
                half = [wx / 2.0, wy / 2.0, h / 2.0]
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half,
                    physicsClientId=self.CLIENT)
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half,
                    rgbaColor=[0.55, 0.35, 0.20, 0.45],
                    physicsClientId=self.CLIENT)
                bid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[ox, oy, h / 2.0],
                    physicsClientId=self.CLIENT)
                self._obstacle_ids.append(bid)
            except Exception:
                pass

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
        """Place fading visual-only spheres along the recent flight path."""
        try:
            n = self.TRAIL_MARKERS
            if n <= 0:
                return
            if not self._trail_bodies:
                for i in range(n):
                    # Older points are small and faint; recent points are more
                    # visible, so the motion direction is readable at a glance.
                    t = 0.0 if n <= 1 else i / float(n - 1)
                    radius = 0.010 + 0.014 * t
                    alpha = 0.15 + 0.55 * t
                    vis = p.createVisualShape(
                        p.GEOM_SPHERE, radius=radius,
                        rgbaColor=[0.18, 0.55, 1.0, alpha],
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

    def _update_drone_marker(self, drone_pos, yaw):
        """Place a subtle halo and heading cue on the drone."""
        try:
            pos = np.asarray(drone_pos, dtype=np.float32)
            if self._drone_marker_id is None:
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.032,
                    rgbaColor=[0.10, 0.85, 0.95, 0.45],
                    physicsClientId=self.CLIENT)
                self._drone_marker_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=(pos + [0.0, 0.0, 0.035]).tolist(),
                    physicsClientId=self.CLIENT)
            else:
                p.resetBasePositionAndOrientation(
                    self._drone_marker_id,
                    (pos + [0.0, 0.0, 0.035]).tolist(), [0, 0, 0, 1],
                    physicsClientId=self.CLIENT)
            self._update_heading_marker(pos, float(yaw))
        except Exception:
            pass

    def _update_heading_marker(self, drone_pos, yaw):
        """Draw a short visual-only marker in front of the drone nose."""
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        zoff = np.array([0.0, 0.0, 0.08], dtype=np.float32)
        shaft_len = 0.26
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        shaft_pos = drone_pos + zoff + direction * (0.5 * shaft_len)
        tip_pos = drone_pos + zoff + direction * shaft_len
        try:
            if self._heading_marker_id is None:
                shaft_vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.5 * shaft_len, 0.010, 0.010],
                    rgbaColor=[0.95, 1.0, 1.0, 0.80],
                    physicsClientId=self.CLIENT)
                self._heading_marker_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=shaft_vis,
                    basePosition=shaft_pos.tolist(),
                    baseOrientation=quat,
                    physicsClientId=self.CLIENT)
                tip_vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.026,
                    rgbaColor=[1.0, 0.95, 0.25, 0.90],
                    physicsClientId=self.CLIENT)
                self._heading_tip_id = p.createMultiBody(
                    baseMass=0, baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=tip_vis,
                    basePosition=tip_pos.tolist(),
                    physicsClientId=self.CLIENT)
            else:
                p.resetBasePositionAndOrientation(
                    self._heading_marker_id, shaft_pos.tolist(), quat,
                    physicsClientId=self.CLIENT)
                p.resetBasePositionAndOrientation(
                    self._heading_tip_id, tip_pos.tolist(), [0, 0, 0, 1],
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
        self._update_drone_marker(drone_pos, s[9])

        # Frame both the drone and the goal: look at their midpoint and back
        # the camera off proportionally to their separation. A moderately
        # high (but not fully top-down) pitch keeps the (semi-transparent) 3 m
        # obstacle pillars from occluding the drone while preserving enough of
        # an oblique angle for a clear, readable 3D view.
        target = (0.5 * (drone_pos + goal)).tolist()
        sep = float(np.linalg.norm(drone_pos - goal))
        distance = float(np.clip(2.4 + 0.9 * sep, 3.0, 9.0))
        self._cam_yaw = (self._cam_yaw + 0.35) % 360.0

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target, distance=distance,
            yaw=self._cam_yaw, pitch=-50.0, roll=0.0, upAxisIndex=2,
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

        # Static obstacle pillars as filled brown rectangles (box footprints).
        if self._obstacle_specs:
            obs_color = np.array([140, 100, 70], dtype=np.uint8)
            for (ox, oy, wx, wy, _h) in self._obstacle_specs:
                x0px, y0px = self._world_to_pixel(float(ox - wx / 2.0),
                                                  float(oy - wy / 2.0))
                x1px, y1px = self._world_to_pixel(float(ox + wx / 2.0),
                                                  float(oy + wy / 2.0))
                xa, xb = sorted((x0px, x1px))
                ya, yb = sorted((y0px, y1px))
                image[ya:yb + 1, xa:xb + 1] = obs_color

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
