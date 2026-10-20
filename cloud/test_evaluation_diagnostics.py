"""Recorder contract tests without GPU or simulator dependencies."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'third_party/dreamerv3/embodied/run/diagnostics.py'
spec = importlib.util.spec_from_file_location('diagnostics', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeEnv:
    CURRENT_EVAL_SEED = 10
    TARGET_POS = np.array([5., 0., 1.])
    BOUNDS = [[-5, 5], [-5, 5], [0, 4]]
    _obstacle_specs = [[1, 0, 1, 1, 3]]
    _dynamic_obstacle_specs = []
    _static_obstacle_ids = [1]
    _dynamic_obstacle_ids = []
    DRONE_IDS = [0]
    CLIENT = 0
    CTRL_FREQ, PYB_FREQ, SPEED_LIMIT = 30, 240, 2.
    COLLISION_DISTANCE, LIDAR_RANGE, LIDAR_START_OFFSET = .15, 4., .06
    def __init__(self):
        self.state = np.zeros(20)
        self.state[2] = 1
    def _resample_task(self):
        pass
    def _getDroneStateVector(self, _):
        return self.state
    def step(self, action):
        first = bool(action['reset'])
        if first:
            self.state[0] = 0
        else:
            self.state[0] += action['action'][0]
        return dict(is_first=first, is_last=not first, reward=np.float32(0),
                    state=self.state[:3].copy(), **{'log/last/success': np.float32(1)})


class Tests(unittest.TestCase):
    def test_all_modes_preserve_outputs_and_enforce_quota(self):
        bullet = types.SimpleNamespace(getClosestPoints=lambda *a, **k: [])
        with patch.dict(sys.modules, pybullet=bullet):
            for mode in ['summary', 'full']:
                with tempfile.TemporaryDirectory() as tmp:
                    wrapped = m.DiagnosticEnv(FakeEnv(), tmp, mode, 1, 0)
                    bare = FakeEnv()
                    for reset in [True, False, True, False]:
                        action = dict(reset=reset, action=np.array([1., 0, 0, .5]))
                        actual, expected = wrapped.step(action), bare.step(action)
                        for key in actual:
                            np.testing.assert_array_equal(actual[key], expected[key])
                    episodes = list(Path(tmp).glob('*.episode.json'))
                    self.assertEqual(len(episodes), 1)
                    episode = json.loads(episodes[0].read_text())
                    self.assertEqual(episode['steps'], 1)
                    self.assertEqual(episode['path_length'], 1)
                    self.assertEqual(episode['terminal']['target_velocity'], [1, 0, 0])
                    trajectories = list(Path(tmp).glob('*.trajectory.jsonl'))
                    self.assertEqual(len(trajectories), int(mode == 'full'))
                    if trajectories:
                        rows = [json.loads(x) for x in trajectories[0].read_text().splitlines()]
                        self.assertIsNone(rows[0]['action_executed'])
                        self.assertEqual(rows[1]['state'][0], 1)
                        self.assertEqual(rows[1]['action_executed'], [1, 0, 0, .5])

    def test_event_flags_overlap_and_geometry_is_frozen(self):
        bullet = types.SimpleNamespace(getClosestPoints=lambda *a, **k: [tuple([0]*8 + [-.01])])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, pybullet=bullet):
            env = FakeEnv()
            wrapper = m.DiagnosticEnv(env, tmp, 'full', 1, 0)
            wrapper.step(dict(reset=True, action=np.zeros(4)))
            env.state[2] = 5
            env.state[7] = 1.3
            wrapper.step(dict(reset=False, action=np.zeros(4)))
            result = json.loads(next(Path(tmp).glob('*.episode.json')).read_text())
            self.assertTrue(result['terminal']['events']['above_ceiling'])
            self.assertTrue(result['terminal']['events']['roll_limit'])
            self.assertEqual(result['terminal']['nearby_obstacles'][0]['clearance'], -.01)
            geometry = json.loads(next(Path(tmp).glob('*.map.json')).read_text())
            self.assertEqual(geometry['start'][2], 1)


if __name__ == '__main__':
    unittest.main()
