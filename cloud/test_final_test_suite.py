"""CPU-independent checks for final-test isolation, checkpoint paths, and statistics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from final_test_suite import SCENARIOS, build_manifest, validate_manifest, validate_coverage, summarize, digest
from eval_final import resolve_checkpoint, RecordingEnv, with_test_environment


class FinalSuiteTest(unittest.TestCase):
    def test_legacy_config_missing_dynamic_keys(self):
        saved = {
            'task': 'drone_nav', 'agent': {'encoder': {'units': 128}},
            'batch_size': 32,
            'env': {'drone': {'eval_seed_base': 190000, 'perception': 'split',
                              'lidar_vbeams': 8, 'n_obstacles': 44}},
        }
        original = copy.deepcopy(saved)
        for enabled, motion in ((False, 'boundary'), (True, 'collision_reverse')):
            changes = {'dynamic_obstacle_motion': motion,
                       'dynamic_obstacles_enabled': enabled,
                       'dynamic_obstacle_diameters': [.25, .5, .75, 1.0],
                       'eval_seed_base': 1000000}
            result = with_test_environment(saved, changes)
            self.assertEqual(result['env']['drone']['dynamic_obstacle_motion'], motion)
            self.assertEqual(result['env']['drone']['dynamic_obstacles_enabled'], enabled)
            self.assertEqual(result['agent'], saved['agent'])
            self.assertEqual(result['batch_size'], 32)
            self.assertEqual(result['env']['drone']['perception'], 'split')
            result['env']['drone']['dynamic_obstacle_diameters'].append(2)
            self.assertEqual(len(changes['dynamic_obstacle_diameters']), 4)
            self.assertEqual(saved, original)

    def test_disjoint_and_stable(self):
        used = set(range(100000, 210128))
        for scene in SCENARIOS:
            m = build_manifest(scene)
            self.assertEqual(m, build_manifest(scene))
            validate_manifest(m)
            self.assertEqual(len(set(m['map_seeds'])), 256)
            self.assertFalse(used.intersection(m['map_seeds']))
            used.update(m['map_seeds'])
        self.assertEqual(build_manifest('20_sparse_dynamic')['presets'],
                         ['drone_static_20_sparse', 'drone_dynamic_20_sparse'])

    def test_tamper_and_source_drift(self):
        m = build_manifest('10_dense')
        m['map_seeds'][0] += 1
        with self.assertRaises(ValueError): validate_manifest(m)
        m = build_manifest('10_dense')
        m['sources']['drone_nav/envs/nav_aviary.py'] = 'changed'
        m['sha256'] = digest({k: v for k, v in m.items() if k != 'sha256'})
        with self.assertRaises(ValueError): validate_manifest(m)

    def test_coverage(self):
        m = build_manifest('20_sparse')
        rows = [dict(worker=w, eval_map_slot=s, eval_map_seed=m['seed_base']+s*16+w)
                for w in range(16) for s in range(16)]
        validate_coverage(rows, m)
        for bad in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError): validate_coverage(bad, m)
        bad = copy.deepcopy(rows)
        bad[0]['worker'] = 2
        with self.assertRaises(ValueError): validate_coverage(bad, m)

    def test_summary_denominators(self):
        rows = [dict(success=1, collision=0, crash=0, timeout=0, episode_length=60),
                dict(success=0, collision=1, crash=0, timeout=0, episode_length=30)]
        result = summarize(rows, 30)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['successful_time_seconds_mean'], 2)
        self.assertLess(result['success_ci95'][0], .5)
        self.assertGreater(result['success_ci95'][1], .5)
        rows[0]['success'] = 0
        self.assertIsNone(summarize(rows, 30)['successful_time_seconds_mean'])

    def test_checkpoint_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            gen = root / 'generation'
            gen.mkdir()
            (root / 'latest').write_text('generation')
            for name in ('agent.pkl', 'step.pkl', 'done'):
                (gen / name).touch()
            self.assertEqual(resolve_checkpoint(root), gen)
            self.assertEqual(resolve_checkpoint(gen), gen)
            best = root / 'best.ckpt'
            best.touch()
            self.assertEqual(resolve_checkpoint(best), best)
            (gen / 'done').unlink()
            with self.assertRaises(FileNotFoundError): resolve_checkpoint(root)

    def test_record_before_movement_and_close_delegation(self):
        class Array(list):
            def tolist(self): return list(self)
        class Physics:
            CURRENT_EVAL_SEED = 1
            INIT_XYZS = [Array([0, 0, 1])]
            TARGET_POS = Array([4, 0, 1])
            _obstacle_specs = [[1, 1, 1, 1, 2]]
            _dynamic_obstacle_specs = [[2, 2, .5, 5, .3, 0]]
            def _resample_task(self): pass
        class Adapter:
            env = Physics()
            closed = False
            def step(self, action): return {'is_first': action['reset']}
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            wrapped = RecordingEnv(Adapter(), tmp)
            wrapped.step({'reset': True})
            initial = json.loads((Path(tmp) / '1.json').read_text())
            self.assertEqual(initial['dynamic_cylinders'][0][-2:], [.3, 0])
            wrapped.step({'reset': False})
            wrapped.close()
            self.assertTrue(wrapped.env.closed)


if __name__ == '__main__':
    unittest.main()
