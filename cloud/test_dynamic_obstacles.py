"""Run with python -m unittest discover -s cloud -p test_dynamic_obstacles.py."""
import importlib.util
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location(
    "moving_geometry", Path(__file__).resolve().parents[1] / "drone_nav/moving_geometry.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class MotionTest(unittest.TestCase):
    def test_swept_wall_prevents_tunnelling(self):
        s = [[-2, 0, .25, 5, 10, 0]]
        out = g.advance_cylinders(s, [(0, 0, .1, 3, 5)], (10, 10), .05, 1)
        self.assertEqual(out[0][:2], [-2, 0])
        self.assertEqual(out[0][4], -10)
        self.assertEqual(s[0][4], 10)

    def test_free_motion_and_boundary(self):
        out = g.advance_cylinders([[0, 0, .25, 5, 1, .5]], [], (10, 10), .05, .1)
        self.assertEqual(out[0][:2], [.1, .05])
        out = g.advance_cylinders([[9.7, 0, .25, 5, 1, 0]], [], (10, 10), .05, .1)
        self.assertEqual(out[0][0], 9.7)
        self.assertEqual(out[0][4], -1)

    def test_crossing_peer_sweeps(self):
        states = [[-1, 0, .1, 5, 2, 0], [0, -1, .1, 5, 0, 2]]
        out = g.advance_cylinders(states, [], (10, 10), .05, 1)
        self.assertEqual(out[0][0], 1)
        self.assertEqual(out[1][1], -1)
        self.assertEqual(out[1][5], -2)

    def test_head_on_swap_rejected(self):
        out = g.advance_cylinders([[-1, 0, .2, 5, 2, 0], [1, 0, .2, 5, -2, 0]],
                                 [], (10, 10), .05, 1)
        self.assertEqual([s[0] for s in out], [-1, 1])

    def test_seeded_long_rollout_clearance(self):
        rng = random.Random(7)
        states = [[x, y, .25, 5, rng.uniform(-1, 1), rng.uniform(-1, 1)]
                  for x in (-3, 3) for y in (-3, -1, 1, 3)]
        boxes = [(0, 0, 1, 3, 5)]
        for _ in range(1800):
            states = g.advance_cylinders(states, boxes, (5, 5), .05, 1 / 30)
            for i, s in enumerate(states):
                self.assertLessEqual(abs(s[0]) + s[2], 5)
                self.assertLessEqual(abs(s[1]) + s[2], 5)
                dx, dy = max(abs(s[0]) - .5, 0), max(abs(s[1]) - 1.5, 0)
                self.assertGreaterEqual((dx * dx + dy * dy) ** .5, s[2] + .05 - 1e-9)
                for other in states[i + 1:]:
                    distance = ((s[0]-other[0])**2 + (s[1]-other[1])**2)**.5
                    self.assertGreaterEqual(distance, s[2] + other[2] + .05 - 1e-9)

    def test_segment_parallel_tangent_and_miss(self):
        self.assertTrue(g.segment_hits_box((-2, 1), (2, 1), (-1, 1, -1, 1)))
        self.assertFalse(g.segment_hits_box((-2, 2), (2, 2), (-1, 1, -1, 1)))


if __name__ == "__main__":
    unittest.main()
