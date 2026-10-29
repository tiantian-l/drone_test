"""Map-based reference baseline: online LiDAR occupancy grid + grid shortest-path replanning.

Information available to the controller (``task_info``)
-------------------------------------------------------
* Always: the same observation as the learned policies -- goal displacement,
  world-frame velocity, attitude (yaw used) and the 8x72 LiDAR scan. No
  simulator obstacle geometry, no obstacle velocities.
* ``task_info='region'`` (Map-A*): additionally the flight region B and the goal
  position in the world frame, given once per episode via ``set_task``. The map
  is built in the world frame (p_t = p_g - dp_t), the region boundary is a
  keep-out band, and the ground is at z = 0.
* ``task_info='none'`` (Map-A*, no region): observation only. The map is built
  in a goal-centred frame (p_t = -dp_t); ``set_task`` is ignored. The grid is a
  search window around start and goal (``window_margin``), not the flight
  region, and has no keep-out band. The unknown ground height is estimated from
  LiDAR: when at least ``ground_fraction`` of a downward layer's rays return at
  the same lowest height, that height is taken as ground (running minimum).

Pipeline (every control step, 30 Hz)
-----------------------------------
1. Mapping. 2D log-odds occupancy grid over the flight region (default 0.1 m).
   Every LiDAR return whose hit point lies above the ground filter marks its
   cell as occupied, regardless of height (conservative: no overflight).
   The two near-horizontal layers carve free space along their rays, so cells
   vacated by moving obstacles are cleared again. A ray clears a cell only
   where it passes at or below the highest return seen in that cell; a ray
   passing over a low obstacle therefore does not erase it. Unobserved cells are free
   (optimistic assumption for planning in unknown terrain).
2. Planning (every ``replan_period`` s, or when the current path is blocked).
   Distance transform of the occupied set gives clearance d. Cells with
   d < hard_radius are impassable; cells with d < inflation are passable only
   at a large penalty (so the UAV can leave them); elsewhere the cost decays
   with clearance. A minimum-cost 8-connected path from the UAV to the goal
   cell is computed with Dijkstra's algorithm on this cost grid
   (skimage.graph.MCP_Geometric). With an admissible heuristic, A* returns a
   path of the same cost; Dijkstra is used only because a compiled
   implementation is available.
3. Tracking. Pure pursuit on the path with a look-ahead distance; horizontal
   speed is limited by the cruise speed, the clearance around the UAV and the
   remaining goal distance. Altitude tracks the goal altitude with a
   proportional law. The command change is rate limited as in the APF baseline.
4. Local safety layer (raw current scan, independent of the map). For the
   closest return in each 30-degree sector within safety_radius and within a
   vertical band around the UAV, the commanded velocity component towards it
   is removed (fully at stop_radius); a small outward push away from the single
   nearest return is added, and the speed is never increased by this layer.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric


@dataclass(frozen=True)
class AStarConfig:
    cruise_speed: float = 1.2        # m/s, horizontal speed cap
    inflation: float = 0.45          # m, soft obstacle inflation
    hard_radius: float = 0.3         # m, impassable radius around occupied cells
    clearance_weight: float = 4.0    # extra cost near obstacles
    clearance_scale: float = 0.5     # m, decay length of clearance cost
    lookahead: float = 1.0           # m, pure-pursuit look-ahead
    replan_period: float = 0.2       # s
    resolution: float = 0.1          # m per cell
    boundary_margin: float = 0.3     # m, keep-out band inside the flight region
    slow_clearance: float = 0.8      # m, full speed only beyond this clearance
    min_speed_fraction: float = 0.3
    vertical_gain: float = 1.0       # 1/s
    max_vertical_speed: float = 0.8  # m/s
    max_acceleration: float = 3.0    # m/s^2, command change limit
    ground_filter: float = 0.15      # m, ignore hits below this height
    safety_radius: float = 0.5       # m, local safety layer activation distance
    stop_radius: float = 0.3         # m, approaching velocity fully removed here
    safety_push: float = 0.3         # m/s, outward push at stop_radius
    safety_band: float = 0.4         # m, vertical band of returns considered
    hit_logodds: float = 0.85
    miss_logodds: float = 0.4
    occupied_threshold: float = 0.5
    logodds_clamp: float = 3.5
    window_margin: float = 6.0       # m, search window around start/goal ('none' only)
    ground_fraction: float = 0.25    # ground detection threshold ('none' only)

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'{k} must be finite and nonnegative')
        if self.hard_radius > self.inflation:
            raise ValueError('hard_radius must not exceed inflation')
        for k in ('cruise_speed', 'lookahead', 'replan_period', 'resolution', 'max_acceleration'):
            if getattr(self, k) <= 0:
                raise ValueError(f'{k} must be positive')


class AStarPolicy:
    TASK_INFO = {'region': ('goal_world', 'flight_region'), 'none': ()}

    def __init__(self, *, bounds, lidar_range=4.0, lidar_start_offset=0.06, lidar_hres_deg=10.0,
                 lidar_vfov=(-10.0, 20.0), lidar_vbeams=4, speed_limit=2.0, ctrl_freq=30,
                 include_angular_velocity=False, config=None, task_info='region'):
        if task_info not in self.TASK_INFO:
            raise ValueError(f'task_info must be one of {sorted(self.TASK_INFO)}')
        self.task_info = task_info
        self.PRIVILEGED_TASK_INFO = self.TASK_INFO[task_info]
        self.config = config or AStarConfig()
        self.bounds = np.asarray(bounds, dtype=float).reshape(3, 2)
        self.range, self.offset, self.speed_limit = float(lidar_range), float(lidar_start_offset), float(speed_limit)
        self.dt = 1.0 / ctrl_freq
        self.state_size = 12 if include_angular_velocity else 9
        h = max(1, int(round(360 / lidar_hres_deg)))
        self.azimuth = np.deg2rad(np.arange(h) * lidar_hres_deg)[None, :]
        el = np.linspace(*lidar_vfov, lidar_vbeams) if lidar_vbeams > 1 else [np.mean(lidar_vfov)]
        self.elevation = np.deg2rad(np.asarray(el, dtype=float))[:, None]
        self.scan_shape = (lidar_vbeams, h)
        # Two layers closest to horizontal carve free space.
        order = np.argsort(np.abs(self.elevation[:, 0]))
        self.carve_layers = np.sort(order[:2])
        r = self.config.resolution
        steps = np.arange(0.0, self.range + r, r / 2)
        self.carve_steps = steps  # sample spacing along rays
        k = int(np.ceil(self.config.hard_radius / r))
        di, dj = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing='ij')
        inside = np.hypot(di, dj) * r < self.config.hard_radius
        self._disk = np.stack([di[inside], dj[inside]], axis=1)
        self.goal = None
        self.reset()

    @classmethod
    def from_environment(cls, settings, config=None, task_info='region'):
        keys = ('lidar_range', 'lidar_start_offset', 'lidar_hres_deg', 'lidar_vfov', 'lidar_vbeams',
                'speed_limit', 'ctrl_freq', 'include_angular_velocity')
        if not settings.get('lidar_enabled', True):
            raise ValueError('Planner requires LiDAR')
        return cls(bounds=settings['bounds'], **{k: settings[k] for k in keys if k in settings},
                   config=config, task_info=task_info)

    # ------------------------------------------------------------------ episode
    def _setup_grid(self, x0, x1, y0, y1, keep_out):
        r = self.config.resolution
        self.x0, self.y0 = float(x0), float(y0)
        self.nx = int(np.ceil((x1 - x0) / r))
        self.ny = int(np.ceil((y1 - y0) / r))
        cx = self.x0 + (np.arange(self.nx) + 0.5) * r
        cy = self.y0 + (np.arange(self.ny) + 0.5) * r
        gx, gy = np.meshgrid(cx, cy, indexing='ij')
        if keep_out:
            m = self.config.boundary_margin
            self.outside = (gx < x0 + m) | (gx > x1 - m) | (gy < y0 + m) | (gy > y1 - m)
        else:
            self.outside = np.zeros(gx.shape, dtype=bool)
        self.logodds = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.hit_top = np.full((self.nx, self.ny), -np.inf, dtype=np.float32)
        self._clear = np.full((self.nx, self.ny), np.inf)

    def reset(self):
        self._previous_command = np.zeros(3)
        self.path = None
        self.path_index = 0
        self.since_plan = np.inf
        self.goal = None
        self.ground = 0.0 if self.task_info == 'region' else -np.inf
        self.diagnostics = {}
        if self.task_info == 'region':
            (x0, x1), (y0, y1) = self.bounds[0], self.bounds[1]
            self._setup_grid(x0, x1, y0, y1, keep_out=True)
        else:
            self.nx = self.ny = 0  # built from the first observation

    def set_task(self, goal_world):
        """Declared task information (goal in world frame); ignored with task_info='none'."""
        if self.task_info == 'region':
            self.goal = np.asarray(goal_world, dtype=float).copy()

    def _start_goal_frame(self, state):
        """'none': goal-centred frame and a search window around start and goal."""
        self.goal = np.zeros(3)
        start = -state[:3]
        m = self.config.window_margin
        self._setup_grid(min(0.0, start[0]) - m, max(0.0, start[0]) + m,
                         min(0.0, start[1]) - m, max(0.0, start[1]) + m, keep_out=False)

    def _update_ground(self, pos, scan):
        """Estimate the ground height from rings of equal-height downward returns."""
        dist = self.offset + scan * (self.range - self.offset)
        for layer in np.flatnonzero(self.elevation[:, 0] < np.deg2rad(-3.0)):
            hit = scan[layer] < 1.0
            if hit.sum() < self.config.ground_fraction * scan.shape[1]:
                continue
            hz = pos[2] + np.sin(self.elevation[layer, 0]) * dist[layer][hit]
            low = hz.min()
            if np.sum(hz < low + 0.05) >= self.config.ground_fraction * scan.shape[1]:
                self.ground = low if not np.isfinite(self.ground) else min(self.ground, low)

    # ------------------------------------------------------------------ helpers
    def _cell(self, xy):
        r = self.config.resolution
        i = np.floor((xy[..., 0] - self.x0) / r).astype(int)
        j = np.floor((xy[..., 1] - self.y0) / r).astype(int)
        return i, j

    def _in_grid(self, i, j):
        return (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)

    def _world(self, i, j):
        r = self.config.resolution
        return np.stack([self.x0 + (i + 0.5) * r, self.y0 + (j + 0.5) * r], axis=-1)

    def _update_map(self, pos, yaw, scan):
        cfg = self.config
        az = self.azimuth + yaw
        el = self.elevation
        dist = self.offset + scan * (self.range - self.offset)
        horiz = np.cos(el) * dist
        hx = pos[0] + horiz * np.cos(az)
        hy = pos[1] + horiz * np.sin(az)
        hz = pos[2] + np.sin(el) * dist
        hit = (scan < 1.0) & (hz > self.ground + cfg.ground_filter)
        # Free-space carving with near-horizontal layers (before marking hits).
        for layer in self.carve_layers:
            d = np.where(scan[layer] < 1.0, dist[layer], self.range) * np.cos(el[layer, 0])
            s = self.carve_steps[None, :]
            mask = s < (d[:, None] - cfg.resolution)
            ang = az[0][:, None]  # azimuths are shared by all layers
            px = pos[0] + s * np.cos(ang)
            py = pos[1] + s * np.sin(ang)
            pz = np.broadcast_to(pos[2] + s * np.tan(el[layer, 0]), px.shape)
            i, j = self._cell(np.stack([px[mask], py[mask]], axis=-1))
            ok = self._in_grid(i, j)
            ok[ok] &= pz[mask][ok] <= np.where(np.isfinite(self.hit_top[i[ok], j[ok]]),
                                                 self.hit_top[i[ok], j[ok]], np.inf)
            if ok.any():
                flat = np.unique(i[ok] * self.ny + j[ok])
                self.logodds.flat[flat] -= cfg.miss_logodds
        i, j = self._cell(np.stack([hx[hit], hy[hit]], axis=-1))
        ok = self._in_grid(i, j)
        if ok.any():
            np.maximum.at(self.hit_top, (i[ok], j[ok]), hz[hit][ok])
            flat = np.unique(i[ok] * self.ny + j[ok])
            self.logodds.flat[flat] += cfg.hit_logodds
        np.clip(self.logodds, -cfg.logodds_clamp, cfg.logodds_clamp, out=self.logodds)

    def _clearance(self):
        occ = self.logodds > self.config.occupied_threshold
        if not occ.any():
            return np.full(occ.shape, np.inf), occ
        return ndimage.distance_transform_edt(~occ) * self.config.resolution, occ

    def _plan(self, pos):
        cfg = self.config
        clear, occ = self._clearance()
        cost = 1.0 + cfg.clearance_weight * np.exp(-np.maximum(clear - cfg.inflation, 0.0) / cfg.clearance_scale)
        cost = np.where(clear < cfg.inflation, cost + 50.0, cost)
        cost = np.where(self.outside, cost + 50.0, cost)
        cost = np.where(clear < cfg.hard_radius, np.inf, cost)
        si, sj = self._cell(pos[:2])
        gi, gj = self._cell(self.goal[:2])
        si, sj = int(np.clip(si, 0, self.nx - 1)), int(np.clip(sj, 0, self.ny - 1))
        gi, gj = int(np.clip(gi, 0, self.nx - 1)), int(np.clip(gj, 0, self.ny - 1))
        cost[si, sj] = 1.0          # never trap the planner at its own cell
        if not np.isfinite(cost[gi, gj]):
            cost[gi, gj] = 100.0
        self._clear = clear
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs([(si, sj)], [(gi, gj)])
        if not np.isfinite(cum[gi, gj]):
            self.path = None
            return clear
        cells = np.asarray(mcp.traceback((gi, gj)))
        self.path = self._world(cells[:, 0], cells[:, 1])
        self.path_index = 0
        return clear

    def _path_blocked(self):
        """Cheap check between replans: any occupied cell within hard_radius of
        the next 4 m of path (the full clearance map is refreshed at replans)."""
        if self.path is None:
            return True
        seg = self.path[self.path_index:self.path_index + int(4.0 / self.config.resolution)]
        i, j = self._cell(seg)
        i = (i[:, None] + self._disk[:, 0][None, :]).ravel()
        j = (j[:, None] + self._disk[:, 1][None, :]).ravel()
        ok = self._in_grid(i, j)
        return bool(np.any(self.logodds[i[ok], j[ok]] > self.config.occupied_threshold))

    def _safety(self, velocity, yaw, scan):
        cfg = self.config
        dist = self.offset + scan * (self.range - self.offset)
        horiz = np.cos(self.elevation) * dist
        dz = np.sin(self.elevation) * dist
        valid = (scan < 1.0) & (np.abs(dz) < cfg.safety_band)
        if not valid.any():
            return velocity, None
        horiz = np.where(valid, horiz, np.inf).min(axis=0)  # per azimuth
        nearest = float(horiz.min())
        az = self.azimuth[0] + yaw
        sector = (np.arange(len(az)) * 360 // len(az)) // 30
        out = velocity.copy()
        speed_cap = max(float(np.linalg.norm(velocity)), cfg.safety_push)
        for k in np.unique(sector):
            m = sector == k
            idx = np.flatnonzero(m)[np.argmin(horiz[m])]
            d = horiz[idx]
            if not np.isfinite(d) or d >= cfg.safety_radius:
                continue
            u = np.array([np.cos(az[idx]), np.sin(az[idx]), 0.0])
            scale = float(np.clip((cfg.safety_radius - d) / max(cfg.safety_radius - cfg.stop_radius, 1e-6), 0, 1))
            toward = float(out @ u)
            if toward > 0:
                out -= scale * toward * u
        i0 = int(np.argmin(horiz))
        if horiz[i0] < cfg.safety_radius:  # single outward push from the nearest return
            u0 = np.array([np.cos(az[i0]), np.sin(az[i0]), 0.0])
            s0 = float(np.clip((cfg.safety_radius - horiz[i0]) / max(cfg.safety_radius - cfg.stop_radius, 1e-6), 0, 1))
            out -= s0 * cfg.safety_push * u0
        n = float(np.linalg.norm(out))
        if n > speed_cap:
            out *= speed_cap / n
        return out, nearest

    # ------------------------------------------------------------------ control
    def act(self, obs):
        if isinstance(obs, dict):
            state = np.asarray(obs['state'], dtype=float)
            scan = np.asarray(obs['lidar'], dtype=float)
        else:
            flat = np.asarray(obs, dtype=float)
            state, scan = flat[:self.state_size], flat[self.state_size:]
        scan = scan.reshape(self.scan_shape)
        if self.goal is None:
            if self.task_info == 'region':
                raise RuntimeError('set_task(goal_world) must be called after each reset')
            self._start_goal_frame(state)
        cfg = self.config
        pos = self.goal - state[:3]
        if self.task_info == 'none':
            self._update_ground(pos, scan)
        self._update_map(pos, state[8], scan)
        self.since_plan += self.dt
        clear = None
        if self.since_plan >= cfg.replan_period - 1e-9 or self.path is None:
            clear = self._plan(pos)
            self.since_plan = 0.0
        else:
            clear = self._clear
            if self._path_blocked():
                clear = self._plan(pos)
                self.since_plan = 0.0
        goal_xy = state[:2]
        goal_dist = float(np.linalg.norm(goal_xy))
        if self.path is None or len(self.path) < 2:
            target = self.goal[:2]
            event = 'no_path'
        else:
            # Advance along the path to the closest point, then look ahead.
            seg = self.path[self.path_index:]
            k = int(np.argmin(np.linalg.norm(seg - pos[:2], axis=1)))
            self.path_index += k
            seg = self.path[self.path_index:]
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
            idx = int(np.searchsorted(arc, cfg.lookahead))
            target = seg[min(idx, len(seg) - 1)]
            event = 'track'
        direction = target - pos[:2]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.zeros(2)
        ci, cj = self._cell(pos[:2])
        local_clear = float(clear[int(np.clip(ci, 0, self.nx - 1)), int(np.clip(cj, 0, self.ny - 1))])
        frac = np.clip((local_clear - cfg.hard_radius) / max(cfg.slow_clearance - cfg.hard_radius, 1e-6),
                       cfg.min_speed_fraction, 1.0)
        speed = min(cfg.cruise_speed * frac, max(goal_dist, 0.2) * 1.5, self.speed_limit)
        vxy = direction * speed
        vz = float(np.clip(cfg.vertical_gain * state[2], -cfg.max_vertical_speed, cfg.max_vertical_speed))
        velocity = np.array([vxy[0], vxy[1], vz])
        norm = np.linalg.norm(velocity)
        if norm > self.speed_limit:
            velocity *= self.speed_limit / norm
        delta = velocity - self._previous_command
        velocity = self._previous_command + delta * min(1.0, cfg.max_acceleration * self.dt / max(np.linalg.norm(delta), 1e-12))
        velocity, nearest = self._safety(velocity, state[8], scan)
        self._previous_command = velocity.copy()
        norm = np.linalg.norm(velocity)
        action = np.zeros(4, dtype=np.float32)
        if norm > 1e-10:
            action[:3] = velocity / norm
            action[3] = min(norm, self.speed_limit) / self.speed_limit
        self.diagnostics = dict(event=event, local_clearance=local_clear, nearest_return=nearest,
                                ground=float(self.ground) if np.isfinite(self.ground) else None,
                                path_len=None if self.path is None else int(len(self.path) - self.path_index))
        return action
