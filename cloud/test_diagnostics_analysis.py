import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from test_evaluation_diagnostics import FakeEnv, m
from analyze_diagnostics import analyze
from plot_diagnostic_episode import plot


class AnalysisTest(unittest.TestCase):
    def fixture(self, root):
        directory = root / 'sample__diagnostics'; directory.mkdir()
        env = FakeEnv()
        wrapper = m.DiagnosticEnv(env, directory, 'full', 1, 0)
        with patch.dict('sys.modules', pybullet=types.SimpleNamespace(getClosestPoints=lambda *a, **k: [])):
            wrapper.step(dict(reset=True, action=np.zeros(4)))
            wrapper.step(dict(reset=False, action=np.array([1., 0, 0, .5])))
        path = next(directory.glob('*.episode.json'))
        episode = json.loads(path.read_text())
        for key in ['collision', 'crash', 'timeout', 'static_collision', 'dynamic_collision']:
            episode['outcome'][key] = 0
        path.write_text(json.dumps(episode))
        (directory / 'run.json').write_text(json.dumps(dict(status='complete', mode='full', episodes=1, workers=1, episodes_per_worker=1)))
        (root / 'sample__summary.json').write_text(json.dumps(dict(episodes=1, success_count=1, collision_count=0)))
        return directory

    def test_analysis_and_plot(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); directory = self.fixture(root)
            rows, failures, summary = analyze(root)
            self.assertEqual(len(rows), 1); self.assertEqual(failures, [])
            self.assertIsNone(rows[0]['min_measured_clearance_m'])
            self.assertTrue(summary['reference_summary_checked'])
            output = plot(directory, 10, root / 'plot.png')
            self.assertGreater(output.stat().st_size, 1000)

    def test_missing_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            directory = self.fixture(Path(t))
            next(directory.glob('*.trajectory.jsonl')).unlink()
            with self.assertRaises(ValueError): analyze(directory)

    def test_reference_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); directory = self.fixture(root)
            (root / 'sample__summary.json').write_text(json.dumps(dict(episodes=1, success_count=0)))
            with self.assertRaises(ValueError): analyze(directory)

    def test_overlapping_failure_labels(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); directory = self.fixture(root)
            path = next(directory.glob('*.episode.json')); episode = json.loads(path.read_text())
            episode['outcome'].update(success=0, crash=1, collision=1)
            episode['terminal']['events'].update(above_ceiling=True, roll_limit=True)
            path.write_text(json.dumps(episode))
            trajectory = directory / episode['trajectory']
            rows = [json.loads(x) for x in trajectory.read_text().splitlines()]
            rows[-1]['events'] = episode['terminal']['events']
            trajectory.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (root / 'sample__summary.json').unlink()
            _, failures, summary = analyze(directory)
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]['above_ceiling']); self.assertTrue(failures[0]['roll_limit'])
            self.assertEqual(summary['counts']['collision'], 1)


if __name__ == '__main__': unittest.main()
