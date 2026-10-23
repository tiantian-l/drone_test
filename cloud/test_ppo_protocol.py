"""Fast configuration/protocol tests; only PyYAML is required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from baselines.ppo.config import environment, test_environment, SCENARIOS
from baselines.ppo.protocol import EpisodeLedger
from cloud.final_test_suite import build_manifest, validate_manifest, validate_coverage, summarize


def terminal(worker, slot, base=100, stride=2, success=False):
    seed = base + worker + stride * slot
    return dict(eval_map_seed=seed, eval_map_slot=slot, is_success=success,
                is_crash=False, is_collision=False, is_timeout=not success,
                static_collision=False, dynamic_collision=False, dynamic_count=0,
                final_distance=0.2 if success else 2.0, min_distance=0.2,
                min_lidar_dist=0.5, initial_geometry=dict(seed=seed))


class ConfigTest(unittest.TestCase):
    def test_all_scenarios_keep_high_resolution_and_reward(self):
        for scenario, count, seconds in zip(SCENARIOS, (11, 22, 44, 88, 44), (20, 20, 40, 40, 40)):
            cfg = environment(scenario)
            self.assertEqual(cfg['n_obstacles'], count)
            self.assertEqual(cfg['episode_len_sec'], seconds)
            self.assertEqual((cfg['lidar_vbeams'], cfg['lidar_hres_deg']), (8, 5.0))
            self.assertEqual(cfg['reward_cfg']['goal_bonus'], 40)
            self.assertEqual(cfg['reward_cfg']['collision_penalty'], 35)
            self.assertEqual(cfg['reward_cfg']['action_smooth'], 0)
            self.assertEqual(cfg['dynamic_obstacles_enabled'], scenario.endswith('dynamic'))

    def test_source_config_preserves_resolved_settings_and_rejects_wrong_scene(self):
        cfg = environment('10_sparse')
        cfg['reward_cfg']['goal_bonus'] = 57
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(dict(task='drone_nav', env=dict(drone=cfg))))
            self.assertEqual(environment('10_sparse', path)['reward_cfg']['goal_bonus'], 57)
            with self.assertRaises(ValueError):
                environment('20_sparse', path)

    def test_cross_scene_preserves_perception_and_clears_dynamic_training(self):
        cfg = environment('20_sparse_dynamic')
        cfg['lidar_vbeams'] = 4
        original = copy.deepcopy(cfg)
        result = test_environment(cfg, build_manifest('10_dense'))
        self.assertEqual(cfg, original)
        self.assertEqual(result['lidar_vbeams'], 4)
        self.assertEqual(result['n_obstacles'], 22)
        self.assertFalse(result['dynamic_obstacles_enabled'])
        self.assertEqual(result['dynamic_obstacle_density'], 0)

    def test_manifest_source_mismatch_is_not_bypassed(self):
        manifest = build_manifest('10_sparse')
        validate_manifest(manifest)
        manifest['sources']['drone_nav/envs/nav_aviary.py'] = 'changed'
        from cloud.final_test_suite import digest
        manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_manifest(manifest)


class LedgerTest(unittest.TestCase):
    def test_fast_workers_cannot_displace_slow_workers(self):
        ledger = EpisodeLedger(2, 2, 2, 100)
        ledger.step([True, False], [terminal(0, 0, success=True), {}])
        ledger.step([True, True], [terminal(0, 1), terminal(1, 0)])
        ledger.step([True, False], [{}, {}])  # surplus terminal info deliberately absent
        ledger.step([True, True], [{}, terminal(1, 1, success=True)])
        records, maps = ledger.finish()
        self.assertEqual([r['episode_length'] for r in records], [1, 1, 2, 2])
        self.assertEqual([r['seed'] for r in maps], [100, 101, 102, 103])
        self.assertEqual(summarize(records, 30)['success_rate'], 0.5)
        self.assertAlmostEqual(summarize(records, 30)['successful_time_seconds_mean'], 1.5 / 30)

    def test_full_manifest_coverage(self):
        manifest = build_manifest('20_sparse_dynamic')
        ledger = EpisodeLedger(16, 16, 16, manifest['seed_base'])
        for slot in range(16):
            ledger.step([True] * 16, [terminal(w, slot, manifest['seed_base'], 16) for w in range(16)])
        records, _ = ledger.finish()
        validate_coverage(records, manifest)

    def test_wrong_slot_and_geometry_rejected(self):
        ledger = EpisodeLedger(1, 1, 1, 100)
        with self.assertRaisesRegex(ValueError, 'map order'):
            ledger.step([True], [terminal(0, 1, stride=1)])
        info = terminal(0, 0, stride=1)
        info['initial_geometry']['seed'] = 101
        with self.assertRaisesRegex(ValueError, 'different episode'):
            ledger.step([True], [info])

    def test_incomplete_and_overlapping_streams_rejected(self):
        with self.assertRaises(ValueError):
            EpisodeLedger(16, 8, 8, 100)
        with self.assertRaises(RuntimeError):
            EpisodeLedger(2, 2, 2, 100).finish()


if __name__ == '__main__':
    unittest.main()
