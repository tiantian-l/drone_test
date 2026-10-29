"""Unit tests for the map-based planner baseline (no simulator required)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.astar.policy import AStarConfig, AStarPolicy  # noqa: E402

BOUNDS = [-6.5, 6.5, -5.0, 5.0, 0.05, 4.5]
GEOM = dict(lidar_range=4.0, lidar_start_offset=0.06, lidar_hres_deg=5.0,
            lidar_vfov=(-10.0, 20.0), lidar_vbeams=8, speed_limit=2.0, ctrl_freq=30)


def make(**cfg):
    return AStarPolicy(bounds=BOUNDS, config=AStarConfig(**cfg), **GEOM)


def obs_at(policy, pos, scan, yaw=0.0):
    state = np.zeros(9)
    state[:3] = policy.goal - np.asarray(pos)
    state[8] = yaw
    return dict(state=state, lidar=scan)


def wall_scan(policy, pos, wall_x, height=4.5):
    """Exact scan of an infinite wall x = wall_x of given height (synthetic)."""
    az = policy.azimuth[0]
    el = policy.elevation[:, 0]
    scan = np.ones(policy.scan_shape)
    for li, e in enumerate(el):
        for ai, a in enumerate(az):
            dx = np.cos(e) * np.cos(a)
            if dx <= 1e-6:
                continue
            t = (wall_x - pos[0]) / dx
            z = pos[2] + t * np.sin(e)
            if 0 < t < policy.range and 0.15 < z < height:
                scan[li, ai] = (t - policy.offset) / (policy.range - policy.offset)
    return scan


def test_config_validation():
    with pytest.raises(ValueError):
        AStarConfig(hard_radius=0.5, inflation=0.4)
    with pytest.raises(ValueError):
        AStarConfig(cruise_speed=0.0)


def test_requires_task():
    p = make()
    with pytest.raises(RuntimeError):
        p.act(dict(state=np.zeros(9), lidar=np.ones(p.scan_shape)))


def test_free_space_goes_to_goal():
    p = make()
    p.set_task([5.0, 0.0, 1.0])
    a = None
    for _ in range(40):
        a = p.act(obs_at(p, [-5.0, 0.0, 1.0], np.ones(p.scan_shape)))
    assert a.shape == (4,) and np.all(np.abs(a) <= 1.0 + 1e-6)
    assert a[0] > 0.9 and a[3] > 0


def test_wall_is_mapped_and_avoided():
    p = make()
    p.set_task([5.0, 0.0, 1.0])
    pos = [-2.0, 0.0, 1.0]
    scan = wall_scan(p, pos, 0.0)
    p.act(obs_at(p, pos, scan))
    occ = p.logodds > p.config.occupied_threshold
    i, _ = p._cell(np.array([0.05, 0.0]))
    assert occ[i - 1:i + 2].any(), 'wall cells must be occupied'
    assert np.isfinite(p._clear).any()


def test_carving_keeps_low_obstacle():
    """A horizontal ray passing over a low obstacle must not erase it."""
    p = make()
    p.set_task([5.0, 0.0, 3.0])
    low = wall_scan(p, [-2.0, 0.0, 3.0], 0.0, height=2.8)  # top below UAV
    for _ in range(5):
        p.act(obs_at(p, [-2.0, 0.0, 3.0], low))
    i, j = p._cell(np.array([0.05, 0.0]))
    before = p.logodds[i - 1:i + 2, j].max()
    assert before > p.config.occupied_threshold
    # The horizontal layers see nothing (they pass over); obstacle must persist.
    for _ in range(20):
        p.act(obs_at(p, [-2.0, 0.0, 3.0], low))
    assert p.logodds[i - 1:i + 2, j].max() > p.config.occupied_threshold


def test_moving_obstacle_is_cleared():
    p = make()
    p.set_task([5.0, 0.0, 1.0])
    pos = [-2.0, 0.0, 1.0]
    p.act(obs_at(p, pos, wall_scan(p, pos, 0.0)))
    i, j = p._cell(np.array([0.05, 0.0]))
    assert p.logodds[i - 1:i + 2, j].max() > p.config.occupied_threshold
    for _ in range(20):  # wall gone, near-horizontal rays now pass freely
        p.act(obs_at(p, pos, np.ones(p.scan_shape)))
    assert p.logodds[i - 1:i + 2, j].max() <= p.config.occupied_threshold


def test_safety_layer_removes_approach():
    p = make()
    p.set_task([5.0, 0.0, 1.0])
    pos = [-0.28, 0.0, 1.0]  # inside stop_radius (0.3 m)
    v, nearest = p._safety(np.array([1.0, 0.0, 0.0]), 0.0, wall_scan(p, pos, 0.0))
    assert nearest < 0.3
    assert v[0] < 0.0  # approach removed and pushed away
    pos = [-0.4, 0.0, 1.0]  # partial reduction between stop and safety radius
    v, _ = p._safety(np.array([1.0, 0.0, 0.0]), 0.0, wall_scan(p, pos, 0.0))
    assert 0.0 < v[0] < 1.0


def test_reset_clears_map():
    p = make()
    p.set_task([5.0, 0.0, 1.0])
    pos = [-2.0, 0.0, 1.0]
    p.act(obs_at(p, pos, wall_scan(p, pos, 0.0)))
    p.reset()
    assert not (p.logodds != 0).any() and p.path is None and p.goal is None


# ---------------------------------------------------------------- task_info='none'
def make_none(**cfg):
    return AStarPolicy(bounds=BOUNDS, config=AStarConfig(**cfg), task_info='none', **GEOM)


def test_none_ignores_task_and_builds_goal_frame():
    p = make_none()
    p.set_task([5.0, 0.0, 1.0])
    assert p.goal is None and p.PRIVILEGED_TASK_INFO == ()
    state = np.zeros(9)
    state[:3] = [10.0, 2.0, 0.5]  # goal displacement
    a = p.act(dict(state=state, lidar=np.ones(p.scan_shape)))
    assert np.allclose(p.goal, 0.0)
    assert a[0] > 0.9
    m = p.config.window_margin
    assert np.isclose(p.x0, -10.0 - m) and p.nx == int(np.ceil((10.0 + 2 * m) / p.config.resolution))
    assert not p.outside.any()


def ground_scan(policy, height):
    """Flat ground at `height` below the UAV, nothing else."""
    scan = np.ones(policy.scan_shape)
    for li, e in enumerate(policy.elevation[:, 0]):
        if e < 0:
            t = height / np.sin(-e)
            if t < policy.range:
                scan[li, :] = (t - policy.offset) / (policy.range - policy.offset)
    return scan


def test_none_ground_ring_is_not_an_obstacle():
    p = make_none()
    state = np.zeros(9)
    state[:3] = [8.0, 0.0, -0.1]  # UAV 0.5 m above ground, unknown to the planner
    for _ in range(5):
        p.act(dict(state=state, lidar=ground_scan(p, 0.5)))
    assert np.isfinite(p.ground)
    assert not (p.logodds > p.config.occupied_threshold).any()


def test_region_mode_marks_without_estimation():
    p = make()
    p.set_task([5.0, 0.0, 0.5])
    assert p.ground == 0.0 and p.PRIVILEGED_TASK_INFO == ('goal_world', 'flight_region')
