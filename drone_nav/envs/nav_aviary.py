"""A->B point navigation environment for a single quadrotor.

Built on top of gym-pybullet-drones `BaseRLAviary` using the *velocity*
controller (`ActionType.VEL`).  The observation is purely proprioceptive
(state-only), as requested:

    obs = [
        rel_x, rel_y, rel_z,     # goal_pos - drone_pos      (3)
        vx, vy, vz,              # linear velocity (world)   (3)
        roll, pitch, yaw,        # attitude (euler)          (3)
        wx, wy, wz,              # body angular velocity      (3, optional)
        # for each of the K nearest static obstacles (only if num_obstacles>0):
        ox, oy, oz, radius,      # obstacle_center - drone, radius  (4 each)
    ]

Static obstacles are tall cylindrical pillars spanning the flight volume, so
the task reduces to 2-D avoidance. They are resampled every episode (avoiding
the start and goal) and exposed to the agent as the K nearest relative vectors.

Action (4-dim, ActionType.VEL):
    a = [dx, dy, dz, speed]
    * (dx, dy, dz) is a (un-normalized) desired velocity direction, the
      DSL-PID velocity controller normalizes it to a unit vector.
    * speed in [-1, 1] -> |speed| scales the target speed up to SPEED_LIMIT.
"""
from collections import deque

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
                 # ---- static obstacles -------------------------------------
                 num_obstacles: int = 0,
                 randomize_obstacles: bool = True,
                 obstacle_positions=None,
                 obstacle_radius_range=(0.15, 0.35),
                 obstacle_sample_range=((-2.0, 2.0), (-2.0, 2.0)),
                 obstacle_clearance: float = 0.6,
                 obstacle_min_gap: float = 0.3,
                 obstacle_margin: float = 0.4,
                 drone_collision_radius: float = 0.12,
                 # placement strategy: "corridor" (between start & goal),
                 # "grid" (density-based whole-map fill), "uniform" (random),
                 # or "mixed" (grid background + guaranteed corridor blockers).
                 obstacle_placement: str = "corridor",
                 corridor_half_width: float = 0.8,
                 corridor_t_range=(0.15, 0.85),
                 corridor_blockers: int = 2,
                 obstacle_density: float = 0.0,
                 # ---- obstacle perception ----------------------------------
                 # "nearest" -> K nearest obstacle relative vectors (legacy);
                 # "lidar"   -> fixed-size 2-D range scan (count-independent);
                 # "none"    -> no obstacle channel in the observation.
                 obstacle_obs_mode: str = "nearest",
                 lidar_num_beams: int = 16,
                 lidar_max_range: float = 2.0,
                 lidar_body_frame: bool = False,
                 # ---- logging / visualization ------------------------------
                 log_video: bool = False,
                 video_size=(384, 384),
                 trail_length: int = 80,
                 # "camera" -> isometric 3-D view (shows altitude); the task is
                 # 3-D so this is the default. "schematic" -> flat top-down.
                 render_mode: str = "camera",
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

        # Static obstacle configuration ---------------------------------------
        self.NUM_OBSTACLES = int(num_obstacles)
        self.RANDOMIZE_OBSTACLES = bool(randomize_obstacles)
        self._fixed_obstacles = (None if obstacle_positions is None
                                 else np.array(obstacle_positions, dtype=np.float32))
        self.OBSTACLE_RADIUS_RANGE = (float(obstacle_radius_range[0]),
                                      float(obstacle_radius_range[1]))
        self.OBSTACLE_SAMPLE_RANGE = np.array(obstacle_sample_range, dtype=np.float32)
        self.OBSTACLE_CLEARANCE = float(obstacle_clearance)
        self.OBSTACLE_MIN_GAP = float(obstacle_min_gap)
        self.OBSTACLE_MARGIN = float(obstacle_margin)
        self.OBSTACLE_PLACEMENT = str(obstacle_placement).lower()
        self.CORRIDOR_HALF_WIDTH = float(corridor_half_width)
        self.CORRIDOR_T_RANGE = (float(corridor_t_range[0]), float(corridor_t_range[1]))
        self.CORRIDOR_BLOCKERS = int(corridor_blockers)
        self.OBSTACLE_DENSITY = float(obstacle_density)
        self.DRONE_COLLISION_RADIUS = float(drone_collision_radius)
        # Obstacle perception ------------------------------------------------
        self.OBSTACLE_OBS_MODE = str(obstacle_obs_mode).lower()
        self.LIDAR_NUM_BEAMS = int(lidar_num_beams)
        self.LIDAR_MAX_RANGE = float(lidar_max_range)
        self.LIDAR_BODY_FRAME = bool(lidar_body_frame)
        # Cache the fixed beam angles (world frame); rotated by yaw at runtime
        # only when ``lidar_body_frame`` is set.
        self._lidar_base_angles = np.linspace(
            0.0, 2.0 * np.pi, self.LIDAR_NUM_BEAMS, endpoint=False
        ).astype(np.float32)
        self._lidar_last = np.ones(self.LIDAR_NUM_BEAMS, dtype=np.float32)
        # Filled by `_sample_obstacles()`; (N, 3) centers and (N,) radii.
        self.OBSTACLE_POS = np.zeros((0, 3), dtype=np.float32)
        self.OBSTACLE_RADII = np.zeros((0,), dtype=np.float32)
        self._obstacle_body_ids = []
        # Placeholder value for empty obstacle slots in the observation.
        self._obs_pad = float(np.max(self.BOUNDS[:, 1] - self.BOUNDS[:, 0]))

        # Visualization (third-person RGB frames for DreamerV3 log/image) -----
        self.LOG_VIDEO = bool(log_video)
        self.VIDEO_SIZE = (int(video_size[0]), int(video_size[1]))  # (H, W)
        self.RENDER_MODE = str(render_mode).lower()
        # Lazily-built isometric camera cache (scale/offsets), see `_iso_setup`.
        self._iso_ready = False
        self.TRAIL_LENGTH = int(trail_length)
        self._trail = deque(maxlen=self.TRAIL_LENGTH)

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
            "obstacle_penalty": 0.5,   # per-step penalty when within OBSTACLE_MARGIN of an obstacle surface
            "collision_penalty": 10.0, # one-off penalty when colliding with an obstacle
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

        self._sample_obstacles()

    def _sample_obstacles(self):
        """Place static cylindrical pillars, avoiding the start and goal.

        Must run AFTER the goal/start are set (so we can keep clearance from
        both) and BEFORE `super().reset()` triggers `_addObstacles()`. Pillars
        span the full vertical flight volume, so avoidance is effectively 2-D.

        Placement strategy (``obstacle_placement``):
          * ``corridor`` - obstacles sampled *between* start and goal, with
            lateral jitter, so they reliably block the direct path.
          * ``grid``     - jittered grid covering the whole sampling area at a
            given density, so obstacles appear everywhere (not just one region).
          * ``uniform``  - fully random placement within the sampling range.
          * ``mixed``    - density grid background PLUS a few guaranteed
            corridor blockers, so the whole map is populated yet the direct
            start->goal path is always obstructed (no trivial straight shots).
        """
        self.OBSTACLE_POS = np.zeros((0, 3), dtype=np.float32)
        self.OBSTACLE_RADII = np.zeros((0,), dtype=np.float32)
        n = self.NUM_OBSTACLES
        placement = self.OBSTACLE_PLACEMENT
        want_grid = placement == "grid"
        want_mixed = placement == "mixed"
        has_grid_fill = (want_grid or want_mixed) and self.OBSTACLE_DENSITY > 0
        has_blockers = want_mixed and self.CORRIDOR_BLOCKERS > 0
        if n <= 0 and not has_grid_fill and not has_blockers:
            return

        # Fixed layout (optionally with per-obstacle radius in a 4th column).
        if self._fixed_obstacles is not None and not self.RANDOMIZE_OBSTACLES:
            fo = self._fixed_obstacles.reshape(-1, self._fixed_obstacles.shape[-1])
            pos = fo[:, :2]
            if fo.shape[1] >= 3:
                radii = fo[:, 2]
            else:
                radii = np.full(len(pos), float(np.mean(self.OBSTACLE_RADIUS_RANGE)))
            centers = np.column_stack([pos, np.full(len(pos), self.BOUNDS[2, 1] * 0.5)])
            self.OBSTACLE_POS = centers.astype(np.float32)
            self.OBSTACLE_RADII = radii.astype(np.float32)
            return

        start_xy = np.asarray(self.INIT_XYZS[0, :2], dtype=np.float32)
        goal_xy = np.asarray(self.TARGET_POS[:2], dtype=np.float32)
        z_center = float(self.BOUNDS[2, 1] * 0.5)

        if want_grid:
            positions, radii = self._sample_grid(start_xy, goal_xy)
        elif want_mixed:
            positions, radii = self._sample_mixed(start_xy, goal_xy)
        elif self.OBSTACLE_PLACEMENT == "uniform":
            positions, radii = self._sample_uniform(start_xy, goal_xy)
        else:  # "corridor" (default)
            positions, radii = self._sample_corridor(start_xy, goal_xy)

        if positions:
            self.OBSTACLE_POS = np.array(
                [[c[0], c[1], z_center] for c in positions], dtype=np.float32)
            self.OBSTACLE_RADII = np.array(radii, dtype=np.float32)

    def _accept(self, c, r, start_xy, goal_xy, positions, radii):
        """Reject candidates too close to start/goal or other obstacles."""
        if np.linalg.norm(c - start_xy) < r + self.OBSTACLE_CLEARANCE:
            return False
        if np.linalg.norm(c - goal_xy) < r + self.OBSTACLE_CLEARANCE:
            return False
        if any(np.linalg.norm(c - pc) < r + pr + self.OBSTACLE_MIN_GAP
               for pc, pr in zip(positions, radii)):
            return False
        return True

    def _sample_uniform(self, start_xy, goal_xy):
        (xl, xh), (yl, yh) = self.OBSTACLE_SAMPLE_RANGE
        positions, radii = [], []
        max_attempts = max(200, self.NUM_OBSTACLES * 200)
        for _ in range(max_attempts):
            if len(positions) >= self.NUM_OBSTACLES:
                break
            r = float(self.np_random.uniform(*self.OBSTACLE_RADIUS_RANGE))
            c = np.array([self.np_random.uniform(xl, xh),
                          self.np_random.uniform(yl, yh)], dtype=np.float32)
            if self._accept(c, r, start_xy, goal_xy, positions, radii):
                positions.append(c)
                radii.append(r)
        return positions, radii

    def _sample_corridor(self, start_xy, goal_xy, n=None, positions=None, radii=None):
        """Spread ``n`` obstacles along the start->goal segment with lateral
        jitter. ``positions``/``radii`` may seed already-placed obstacles so
        this can extend an existing layout (used by the ``mixed`` strategy)."""
        (xl, xh), (yl, yh) = self.OBSTACLE_SAMPLE_RANGE
        n = self.NUM_OBSTACLES if n is None else int(n)
        positions = [] if positions is None else positions
        radii = [] if radii is None else radii
        d = goal_xy - start_xy
        length = float(np.linalg.norm(d))
        if length < 1e-3:
            # Degenerate (start == goal): fall back to uniform placement.
            return self._sample_uniform(start_xy, goal_xy)
        u = d / length                      # along-path unit vector
        perp = np.array([-u[1], u[0]], dtype=np.float32)  # lateral unit vector
        t_lo, t_hi = self.CORRIDOR_T_RANGE
        # Evenly spaced anchors along the path so obstacles don't clump.
        anchors = (np.linspace(t_lo, t_hi, n) if n > 1
                   else np.array([(t_lo + t_hi) * 0.5]))
        for t0 in anchors:
            placed = False
            for _ in range(200):
                r = float(self.np_random.uniform(*self.OBSTACLE_RADIUS_RANGE))
                # Jitter along the path (within the per-anchor band) and laterally.
                span = (t_hi - t_lo) / max(n, 1)
                t = float(np.clip(t0 + self.np_random.uniform(-0.5, 0.5) * span,
                                  t_lo, t_hi))
                lat = float(self.np_random.uniform(-self.CORRIDOR_HALF_WIDTH,
                                                   self.CORRIDOR_HALF_WIDTH))
                c = start_xy + t * d + lat * perp
                c = np.array([np.clip(c[0], xl, xh), np.clip(c[1], yl, yh)],
                             dtype=np.float32)
                if self._accept(c, r, start_xy, goal_xy, positions, radii):
                    positions.append(c)
                    radii.append(r)
                    placed = True
                    break
            if not placed:
                continue
        return positions, radii

    def _sample_grid(self, start_xy, goal_xy, positions=None, radii=None):
        """Jittered grid covering the whole sampling area at a given density.

        If ``obstacle_density`` (obstacles per m^2) is set, the grid spacing is
        derived from it; otherwise the grid is sized to fit ``num_obstacles``.
        ``positions``/``radii`` may seed already-placed obstacles so this can
        fill *around* them (used by the ``mixed`` strategy).
        """
        (xl, xh), (yl, yh) = self.OBSTACLE_SAMPLE_RANGE
        positions = [] if positions is None else positions
        radii = [] if radii is None else radii
        area = max((xh - xl) * (yh - yl), 1e-6)
        if self.OBSTACLE_DENSITY > 0:
            spacing = float(1.0 / np.sqrt(self.OBSTACLE_DENSITY))
            cap = None
        else:
            # Choose spacing so the grid roughly yields NUM_OBSTACLES cells.
            spacing = float(np.sqrt(area / max(self.NUM_OBSTACLES, 1)))
            cap = self.NUM_OBSTACLES
        nx = max(int(np.round((xh - xl) / spacing)), 1)
        ny = max(int(np.round((yh - yl) / spacing)), 1)
        # Center the grid within the range.
        ox = xl + ((xh - xl) - (nx - 1) * spacing) * 0.5 if nx > 1 else (xl + xh) * 0.5
        oy = yl + ((yh - yl) - (ny - 1) * spacing) * 0.5 if ny > 1 else (yl + yh) * 0.5
        jitter = spacing * 0.3
        cells = [(ox + i * spacing, oy + j * spacing)
                 for i in range(nx) for j in range(ny)]
        # Shuffle so an optional cap selects a spatially spread subset.
        perm = self.np_random.permutation(len(cells))
        cells = [cells[k] for k in perm]
        # ``cap`` counts only the grid cells we add (seed blockers don't count).
        target = None if cap is None else cap + len(positions)
        for cx, cy in cells:
            if target is not None and len(positions) >= target:
                break
            r = float(self.np_random.uniform(*self.OBSTACLE_RADIUS_RANGE))
            c = np.array([cx + self.np_random.uniform(-jitter, jitter),
                          cy + self.np_random.uniform(-jitter, jitter)],
                         dtype=np.float32)
            c = np.array([np.clip(c[0], xl, xh), np.clip(c[1], yl, yh)],
                         dtype=np.float32)
            if self._accept(c, r, start_xy, goal_xy, positions, radii):
                positions.append(c)
                radii.append(r)
        return positions, radii

    def _sample_mixed(self, start_xy, goal_xy):
        """Guaranteed corridor blockers + a whole-map density grid background.

        First place ``corridor_blockers`` pillars on the direct start->goal
        line (so the path is never trivially clear), then fill the rest of the
        arena with the density grid, keeping clearance from the blockers. This
        removes the "obstacles ended up off the path" failure mode of pure grid
        placement while still teaching general, whole-map avoidance.
        """
        positions, radii = self._sample_corridor(
            start_xy, goal_xy, n=self.CORRIDOR_BLOCKERS)
        if self.OBSTACLE_DENSITY > 0 or self.NUM_OBSTACLES > 0:
            positions, radii = self._sample_grid(
                start_xy, goal_xy, positions=positions, radii=radii)
        return positions, radii

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
        return obs, info

    ################################################################################
    # Observation
    ################################################################################

    def _observationSpace(self):
        dim = 12 if self.INCLUDE_ANG_VEL else 9
        dim += self._obstacle_obs_dim()
        hi = np.inf * np.ones(dim, dtype=np.float32)
        return spaces.Box(low=-hi, high=hi, shape=(dim,), dtype=np.float32)

    def _obstacle_obs_dim(self):
        """Width of the obstacle-perception slice of the observation vector."""
        if self.OBSTACLE_OBS_MODE == "lidar":
            return self.LIDAR_NUM_BEAMS
        if self.OBSTACLE_OBS_MODE == "nearest":
            return 4 * self.NUM_OBSTACLES
        return 0

    def _obstacle_features(self, drone_pos):
        """K nearest obstacles as [rel_x, rel_y, rel_z, radius], padded to K."""
        K = self.NUM_OBSTACLES
        if K <= 0:
            return np.zeros((0,), dtype=np.float32)
        feats = []
        if self.OBSTACLE_POS.shape[0] > 0:
            rel = self.OBSTACLE_POS - drone_pos          # (N, 3)
            order = np.argsort(np.linalg.norm(rel[:, :2], axis=1))
            for i in order[:K]:
                feats.append(np.array(
                    [rel[i, 0], rel[i, 1], rel[i, 2], self.OBSTACLE_RADII[i]],
                    dtype=np.float32))
        # Pad empty slots with a far placeholder and zero radius.
        while len(feats) < K:
            feats.append(np.array([self._obs_pad, self._obs_pad, 0.0, 0.0],
                                  dtype=np.float32))
        return np.concatenate(feats).astype(np.float32)

    def _lidar_scan(self, drone_pos, yaw):
        """Fixed-size 2-D range scan, analytic ray-casting against the
        cylindrical pillars and the arena walls.

        Returns ``LIDAR_NUM_BEAMS`` normalized clearances in ``[0, 1]`` where
        ``1`` means "nothing within ``LIDAR_MAX_RANGE``" (free) and ``0`` means
        the beam is touching a surface. Because the width is fixed, this works
        unchanged for any obstacle count or density (unlike the nearest-K
        encoding, whose width scales with K).
        """
        B = self.LIDAR_NUM_BEAMS
        angles = self._lidar_base_angles
        if self.LIDAR_BODY_FRAME:
            angles = angles + float(yaw)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (B, 2)
        o = drone_pos[:2].astype(np.float32)
        rng = np.full(B, self.LIDAR_MAX_RANGE, dtype=np.float32)

        # --- ray vs. cylinders (2-D circles) -------------------------------
        if self.OBSTACLE_POS.shape[0] > 0:
            c = self.OBSTACLE_POS[:, :2]                 # (N, 2)
            R = self.OBSTACLE_RADII                       # (N,)
            f = o[None, :] - c                            # (N, 2)
            b = dirs @ f.T                                # (B, N)
            cc = (f * f).sum(axis=1) - R ** 2             # (N,)
            disc = b ** 2 - cc[None, :]                   # (B, N)
            sq = np.sqrt(np.maximum(disc, 0.0))
            t_near = -b - sq
            t_far = -b + sq
            t = np.where(t_near > 1e-6, t_near, t_far)    # nearest hit ahead
            t = np.where((disc >= 0.0) & (t > 1e-6), t, np.inf)
            rng = np.minimum(rng, t.min(axis=1))

        # --- ray vs. arena walls (axis-aligned box) ------------------------
        (xl, xh), (yl, yh), _ = self.BOUNDS
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(dirs[:, 0] > 0, (xh - o[0]) / dirs[:, 0],
                          np.where(dirs[:, 0] < 0, (xl - o[0]) / dirs[:, 0], np.inf))
            ty = np.where(dirs[:, 1] > 0, (yh - o[1]) / dirs[:, 1],
                          np.where(dirs[:, 1] < 0, (yl - o[1]) / dirs[:, 1], np.inf))
        rng = np.minimum(rng, np.minimum(tx, ty))

        rng = np.clip(rng, 0.0, self.LIDAR_MAX_RANGE)
        self._lidar_last = (rng / self.LIDAR_MAX_RANGE).astype(np.float32)
        return self._lidar_last

    def _computeObs(self):
        s = self._getDroneStateVector(0)
        rel_goal = self.TARGET_POS - s[0:3]   # (3,)
        vel = s[10:13]                         # (3,)
        rpy = s[7:10]                          # (3,)
        parts = [rel_goal, vel, rpy]
        if self.INCLUDE_ANG_VEL:
            parts.append(s[13:16])             # body angular velocity
        if self.OBSTACLE_OBS_MODE == "lidar":
            parts.append(self._lidar_scan(s[0:3], s[9]))
        elif self.OBSTACLE_OBS_MODE == "nearest" and self.NUM_OBSTACLES > 0:
            parts.append(self._obstacle_features(s[0:3]))
        return np.concatenate(parts).astype(np.float32)

    ################################################################################
    # Reward
    ################################################################################

    def _distance_to_goal(self):
        s = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.TARGET_POS - s[0:3]))

    def _obstacle_surface_distance(self, pos):
        """Signed horizontal distance from the drone to the nearest pillar
        surface (negative when inside a pillar). Returns +inf if none."""
        if self.OBSTACLE_POS.shape[0] == 0:
            return float("inf")
        d = np.linalg.norm(self.OBSTACLE_POS[:, :2] - pos[:2], axis=1)
        return float(np.min(d - self.OBSTACLE_RADII))

    def _is_collision(self, s):
        if self.OBSTACLE_POS.shape[0] == 0:
            return False
        return self._obstacle_surface_distance(s[0:3]) < self.DRONE_COLLISION_RADIUS

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

        # 6) Obstacle clearance shaping + collision penalty.
        if self.OBSTACLE_POS.shape[0] > 0:
            surf = self._obstacle_surface_distance(s[0:3])
            if self.RW["obstacle_penalty"] and surf < self.OBSTACLE_MARGIN:
                closeness = (self.OBSTACLE_MARGIN - max(surf, 0.0)) / self.OBSTACLE_MARGIN
                reward -= self.RW["obstacle_penalty"] * float(np.clip(closeness, 0.0, 1.0))
            if self._is_collision(s):
                reward -= self.RW["collision_penalty"]

        # 7) Crash / out-of-bounds penalty (mirrors _computeTerminated).
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
        if self._is_crash(s) or self._is_collision(s):
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
        collision = self._is_collision(s)
        crash = self._is_crash(s) or collision
        obstacle_dist = self._obstacle_surface_distance(s[0:3])
        return {
            "distance": dist,
            "is_success": bool(success),
            "is_collision": bool(collision),
            "obstacle_distance": (0.0 if not np.isfinite(obstacle_dist)
                                  else float(obstacle_dist)),
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
        # Static cylindrical pillars (collision + visual). Sampled in
        # `_resample_task()` before this housekeeping runs.
        self._obstacle_body_ids = []
        height = float(self.BOUNDS[2, 1])
        for center, radius in zip(self.OBSTACLE_POS, self.OBSTACLE_RADII):
            try:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=float(radius), height=height,
                    physicsClientId=self.CLIENT)
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=float(radius), length=height,
                    rgbaColor=[0.45, 0.45, 0.5, 1.0],
                    physicsClientId=self.CLIENT)
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[float(center[0]), float(center[1]), height * 0.5],
                    physicsClientId=self.CLIENT)
                self._obstacle_body_ids.append(body)
            except Exception:
                pass
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
        """Render one third-person RGB frame for the ``log/image`` video.

        Dispatches on ``render_mode``: ``"camera"`` draws an isometric 3-D view
        (the task is 3-D, so altitude is visible), ``"schematic"`` keeps the
        flat top-down diagram.
        """
        if self.RENDER_MODE == "schematic":
            return self._render_schematic()
        return self._render_camera()

    # ------------------------------------------------------------------ 3-D ---
    def _iso_setup(self):
        """Precompute the isometric camera scale/offset so the whole flight
        volume fits the frame. Runs once (depends only on BOUNDS / VIDEO_SIZE)."""
        az = np.deg2rad(40.0)     # azimuth: view from the south-east
        el = np.deg2rad(55.0)     # high (near bird's-eye) so tall pillars stay
                                  # short on screen and never bury the drone
        self._iso_el = el
        d = np.array([np.cos(el) * np.cos(az),
                      np.cos(el) * np.sin(az), -np.sin(el)], dtype=np.float32)
        right = np.array([-np.sin(az), np.cos(az), 0.0], dtype=np.float32)
        up_cam = np.cross(d, right)
        self._iso_d, self._iso_right, self._iso_up = d, right, up_cam

        (xl, xh), (yl, yh), (zl, zh) = self.BOUNDS
        corners = np.array([[x, y, z] for x in (xl, xh) for y in (yl, yh)
                            for z in (0.0, zh)], dtype=np.float32)
        u = corners @ right
        v = corners @ up_cam
        h, w = self.VIDEO_SIZE
        pad = 0.10
        umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
        scale = (1.0 - 2.0 * pad) * min((w - 1) / max(umax - umin, 1e-6),
                                        (h - 1) / max(vmax - vmin, 1e-6))
        self._iso_scale = float(scale)
        self._iso_offx = float((w - 1) * 0.5 - scale * (umin + umax) * 0.5)
        self._iso_offy = float((h - 1) * 0.5 + scale * (vmin + vmax) * 0.5)
        self._iso_ready = True

    def _iso_project(self, pts):
        """World (..,3) -> (px, py, depth). depth grows into the screen."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        u = pts @ self._iso_right
        v = pts @ self._iso_up
        depth = pts @ self._iso_d
        px = self._iso_offx + self._iso_scale * u
        py = self._iso_offy - self._iso_scale * v
        return px, py, depth

    def _draw_ellipse(self, image, cx, cy, rx, ry, color, alpha=1.0):
        h, w, _ = image.shape
        rx = max(int(rx), 1)
        ry = max(int(ry), 1)
        x0, x1 = max(0, cx - rx), min(w - 1, cx + rx)
        y0, y1 = max(0, cy - ry), min(h - 1, cy + ry)
        if x1 < x0 or y1 < y0:
            return
        yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        region = image[y0:y1 + 1, x0:x1 + 1]
        if alpha >= 1.0:
            region[mask] = color
        else:
            blended = (region[mask].astype(np.float32) * (1.0 - alpha)
                       + color.astype(np.float32) * alpha)
            region[mask] = blended.astype(np.uint8)

    def _fill_rect(self, image, x0, x1, y0, y1, color, alpha=1.0):
        h, w, _ = image.shape
        x0, x1 = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
        if x1 < x0 or y1 < y0:
            return
        if alpha >= 1.0:
            image[y0:y1 + 1, x0:x1 + 1] = color
        else:
            region = image[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
            image[y0:y1 + 1, x0:x1 + 1] = (
                region * (1.0 - alpha) + color.astype(np.float32) * alpha
            ).astype(np.uint8)

    def _render_camera(self):
        """Isometric 3-D view: tall pillars, a drone with a drop-shadow that
        reveals its altitude, the goal, the flown trail and the lidar fan."""
        if not self._iso_ready:
            self._iso_setup()
        h, w = self.VIDEO_SIZE
        image = np.full((h, w, 3), 245, dtype=np.uint8)

        (xl, xh), (yl, yh), (zl, zh) = self.BOUNDS
        H = float(zh)
        s = self._getDroneStateVector(0)
        drone_pos = s[0:3].astype(np.float32)
        goal = self.TARGET_POS.astype(np.float32)
        if len(self._trail) == 0 or np.linalg.norm(self._trail[-1] - drone_pos) > 1e-4:
            self._trail.append(drone_pos.copy())

        # --- ground plane (filled quad) + grid lines -----------------------
        floor = np.array([[xl, yl, 0], [xh, yl, 0], [xh, yh, 0], [xl, yh, 0]],
                         dtype=np.float32)
        fpx, fpy, _ = self._iso_project(floor)
        self._fill_poly(image, fpx, fpy, np.array([224, 228, 236], dtype=np.uint8))
        grid_color = np.array([198, 203, 214], dtype=np.uint8)
        step = 1.0
        gx = np.ceil(xl / step) * step
        while gx <= xh + 1e-6:
            px, py, _ = self._iso_project([[gx, yl, 0], [gx, yh, 0]])
            self._draw_line(image, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), grid_color, 0)
            gx += step
        gy = np.ceil(yl / step) * step
        while gy <= yh + 1e-6:
            px, py, _ = self._iso_project([[xl, gy, 0], [xh, gy, 0]])
            self._draw_line(image, (int(px[0]), int(py[0])), (int(px[1]), int(py[1])), grid_color, 0)
            gy += step

        # --- depth-sort the pillars only (painter's algorithm) ------------
        pillars = []  # (depth, center, radius)
        for c, r in zip(self.OBSTACLE_POS, self.OBSTACLE_RADII):
            _, _, dep = self._iso_project([[c[0], c[1], 0.0]])
            pillars.append((float(dep[0]), c, float(r)))
        # Far (large depth) first so nearer pillars paint over them.
        pillars.sort(key=lambda t: -t[0])

        # --- trail on the ground-projected path (drawn before objects) -----
        if len(self._trail) >= 2:
            trail = np.array(self._trail, dtype=np.float32)
            tpx, tpy, _ = self._iso_project(trail)
            tcol = np.array([52, 120, 220], dtype=np.uint8)
            for i in range(len(trail) - 1):
                self._draw_line(image, (int(tpx[i]), int(tpy[i])),
                                (int(tpx[i + 1]), int(tpy[i + 1])), tcol, 1)

        # Pillars are semi-transparent so the agent stays visible through them.
        for _, c, r in pillars:
            self._draw_pillar(image, c, r, H)

        # The goal and the drone are the task-relevant markers, so always draw
        # them LAST (as an overlay): they are never fully hidden behind a pillar.
        self._draw_marker3d(image, goal, radius=6,
                            color=np.array([230, 55, 55], dtype=np.uint8),
                            core=np.array([255, 220, 220], dtype=np.uint8))
        self._draw_drone3d(image, drone_pos, s)

        return image

    def _fill_poly(self, image, px, py, color):
        """Fill a convex polygon given pixel-space vertex arrays."""
        h, w, _ = image.shape
        ys = py.astype(int)
        y0, y1 = max(0, ys.min()), min(h - 1, ys.max())
        n = len(px)
        for y in range(y0, y1 + 1):
            xs = []
            for i in range(n):
                j = (i + 1) % n
                ya, yb = py[i], py[j]
                if (ya <= y < yb) or (yb <= y < ya):
                    t = (y - ya) / (yb - ya)
                    xs.append(px[i] + t * (px[j] - px[i]))
            if len(xs) >= 2:
                xa, xb = int(min(xs)), int(max(xs))
                self._fill_rect(image, xa, xb, y, y, color)

    def _draw_pillar(self, image, center, radius, height):
        """Draw a cylindrical pillar as a shaded vertical body + top cap."""
        base = [float(center[0]), float(center[1]), 0.0]
        top = [float(center[0]), float(center[1]), float(height)]
        bpx, bpy, _ = self._iso_project([base])
        tpx, tpy, _ = self._iso_project([top])
        cx = int(round(float(bpx[0])))
        by = int(round(float(bpy[0])))
        ty = int(round(float(tpy[0])))
        hw = max(2, int(round(radius * self._iso_scale)))
        rye = max(1, int(round(hw * np.sin(self._iso_el))))
        # Semi-transparent body (so the drone behind stays visible), a brighter
        # shaded edge, a darker base rim and a brighter top cap.
        a = 0.72
        self._fill_rect(image, cx - hw, cx + hw, ty, by,
                        np.array([120, 126, 140], dtype=np.uint8), alpha=a)
        self._fill_rect(image, cx - hw, cx - hw + max(1, hw // 3), ty, by,
                        np.array([150, 156, 168], dtype=np.uint8), alpha=a)
        self._draw_ellipse(image, cx, by, hw, rye,
                           np.array([92, 97, 110], dtype=np.uint8), alpha=a)
        self._draw_ellipse(image, cx, ty, hw, rye,
                           np.array([158, 164, 176], dtype=np.uint8), alpha=a)

    def _draw_marker3d(self, image, pos, radius, color, core):
        """A floating marker with a drop line + ground shadow showing altitude."""
        gpx, gpy, _ = self._iso_project([[pos[0], pos[1], 0.0]])
        apx, apy, _ = self._iso_project([[pos[0], pos[1], pos[2]]])
        gx, gy = int(gpx[0]), int(gpy[0])
        ax, ay = int(apx[0]), int(apy[0])
        self._draw_ellipse(image, gx, gy, max(2, radius - 2),
                           max(1, (radius - 2) // 2), np.array([150, 150, 160], dtype=np.uint8))
        self._draw_line(image, (gx, gy), (ax, ay), np.array([150, 150, 160], dtype=np.uint8), 0)
        self._draw_disc(image, ax, ay, radius, color)
        self._draw_disc(image, ax, ay, max(1, radius // 2), core)

    def _draw_drone3d(self, image, drone_pos, s):
        """Drone marker with altitude drop-line, shadow, heading and lidar fan."""
        gpx, gpy, _ = self._iso_project([[drone_pos[0], drone_pos[1], 0.0]])
        apx, apy, _ = self._iso_project([[drone_pos[0], drone_pos[1], drone_pos[2]]])
        gx, gy = int(gpx[0]), int(gpy[0])
        ax, ay = int(apx[0]), int(apy[0])
        # Lidar fan at the drone's altitude (horizontal beams).
        if self.OBSTACLE_OBS_MODE == "lidar":
            angles = self._lidar_base_angles
            if self.LIDAR_BODY_FRAME:
                angles = angles + float(s[9])
            beam_color = np.array([250, 190, 70], dtype=np.uint8)
            ends = np.stack([
                drone_pos[0] + self._lidar_last * self.LIDAR_MAX_RANGE * np.cos(angles),
                drone_pos[1] + self._lidar_last * self.LIDAR_MAX_RANGE * np.sin(angles),
                np.full(len(angles), drone_pos[2])], axis=1)
            epx, epy, _ = self._iso_project(ends)
            for i in range(len(angles)):
                self._draw_line(image, (ax, ay), (int(epx[i]), int(epy[i])), beam_color, 0)
        # Drop shadow + altitude line.
        self._draw_ellipse(image, gx, gy, 4, 2, np.array([150, 150, 160], dtype=np.uint8))
        self._draw_line(image, (gx, gy), (ax, ay), np.array([120, 120, 135], dtype=np.uint8), 0)
        # Heading arrow (in the horizontal plane at the drone's altitude).
        yaw = float(s[9])
        hpx, hpy, _ = self._iso_project([[drone_pos[0] + 0.35 * np.cos(yaw),
                                          drone_pos[1] + 0.35 * np.sin(yaw),
                                          drone_pos[2]]])
        self._draw_line(image, (ax, ay), (int(hpx[0]), int(hpy[0])),
                        np.array([20, 70, 90], dtype=np.uint8), 1)
        self._draw_disc(image, ax, ay, 6, np.array([35, 200, 210], dtype=np.uint8))
        self._draw_disc(image, ax, ay, 3, np.array([240, 250, 255], dtype=np.uint8))

    # -------------------------------------------------------------- 2-D ---
    def _render_schematic(self):
        h, w = self.VIDEO_SIZE
        image = np.full((h, w, 3), 244, dtype=np.uint8)
        self._draw_grid(image)

        border = np.array([160, 168, 180], dtype=np.uint8)
        image[0:2, :, :] = border
        image[-2:, :, :] = border
        image[:, 0:2, :] = border
        image[:, -2:, :] = border

        # Static obstacles (filled gray discs scaled to pixel radius).
        if self.OBSTACLE_POS.shape[0] > 0:
            (xl, xh), (yl, yh), _ = self.BOUNDS
            pad = 0.08
            span_x = max(xh - xl, 1e-6)
            scale = (1.0 - 2.0 * pad) * (w - 1) / span_x
            obs_fill = np.array([110, 114, 124], dtype=np.uint8)
            for center, radius in zip(self.OBSTACLE_POS, self.OBSTACLE_RADII):
                ox, oy = self._world_to_pixel(float(center[0]), float(center[1]))
                pr = max(2, int(round(float(radius) * scale)))
                self._draw_disc(image, ox, oy, pr, obs_fill)

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
        # Lidar beams (drawn beneath the drone marker so they read as rays).
        if self.OBSTACLE_OBS_MODE == "lidar":
            angles = self._lidar_base_angles
            if self.LIDAR_BODY_FRAME:
                angles = angles + float(s[9])
            beam_color = np.array([250, 190, 70], dtype=np.uint8)
            for ang, norm in zip(angles, self._lidar_last):
                d = float(norm) * self.LIDAR_MAX_RANGE
                ex = float(drone_pos[0]) + d * float(np.cos(float(ang)))
                ey = float(drone_pos[1]) + d * float(np.sin(float(ang)))
                epx, epy = self._world_to_pixel(ex, ey)
                self._draw_line(image, (dx, dy), (epx, epy), beam_color, thickness=0)
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

