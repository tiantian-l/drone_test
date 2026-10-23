"""Actual PyBullet/SB3 integration tests; skipped when dependencies are absent."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/gym-pybullet-drones'))
AVAILABLE = all(importlib.util.find_spec(name) for name in
                ('torch', 'stable_baselines3', 'gymnasium', 'pybullet'))


@unittest.skipUnless(AVAILABLE, 'Install requirements-ppo.txt to run actual training tests')
class IntegrationTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_real_update_validation_and_checkpoint_roundtrip(self):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from baselines.ppo.config import environment
        from baselines.ppo.envs import vector_env
        from baselines.ppo.features import NavigationFeatures
        from baselines.ppo.training import FixedValidation
        cfg = dict(environment=environment('10_sparse'), eval_every=16,
                   validation_workers=1, validation_quota=1, validation_stride=16)
        # A short time limit is test-only; actual scene geometry and sensor remain intact.
        cfg['environment']['episode_len_sec'] = 1
        env = vector_env(cfg['environment'], 1, subprocess=False)
        try:
            model = PPO('MultiInputPolicy', env, n_steps=16, batch_size=8, n_epochs=1,
                        policy_kwargs=dict(features_extractor_class=NavigationFeatures, normalize_images=False),
                        seed=7, device='cpu')
            before = [p.detach().clone() for p in model.policy.parameters()]
            with tempfile.TemporaryDirectory() as directory:
                model.learn(32, callback=FixedValidation(cfg, directory, subprocess=False))
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.policy.parameters())))
                obs = env.reset()
                expected, _ = model.predict(obs, deterministic=True)
                loaded = PPO.load(Path(directory) / 'last_model.zip', device='cpu')
                actual, _ = loaded.predict(obs, deterministic=True)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (1, 1, 4))
                summary = json.loads((Path(directory) / 'validation/step_0000000032/summary.json').read_text())
                self.assertEqual(summary['episodes'], 1)
                self.assertTrue((Path(directory) / 'best_model.zip').is_file())
        finally:
            env.close()

    def test_timeout_terminal_observation_and_geometry_survive_autoreset(self):
        from baselines.ppo.config import environment
        from baselines.ppo.envs import vector_env
        import numpy as np
        cfg = environment('10_sparse')
        cfg['episode_len_sec'] = 0  # first action times out, without altering production code
        env = vector_env(cfg, 1, evaluation=True, quota=2, stride=16, subprocess=False)
        try:
            env.reset()
            _, _, dones, infos = env.step(np.zeros((1, 1, 4), dtype=np.float32))
            self.assertTrue(dones[0])
            self.assertTrue(infos[0]['TimeLimit.truncated'])
            self.assertIn('terminal_observation', infos[0])
            self.assertEqual(infos[0]['initial_geometry']['seed'], cfg['eval_seed_base'])
            self.assertEqual(env.reset_infos[0]['eval_map_seed'], cfg['eval_seed_base'] + 16)
        finally:
            env.close()

    def test_training_seed_matches_dreamer_seed_sequence(self):
        import numpy as np
        from baselines.ppo.config import environment, FRAMEWORK_KEYS
        from baselines.ppo.envs import vector_env
        from drone_nav.envs.nav_aviary import NavigationAviary
        cfg = environment('10_sparse')
        direct = NavigationAviary(**{k: v for k, v in cfg.items() if k not in FRAMEWORK_KEYS})
        env = vector_env(cfg, 1, seed=19, subprocess=False)
        try:
            env.seed(19)  # SB3's seed+index must not replace our SeedSequence mapping
            obs = env.reset()
            worker_seed = int(np.random.SeedSequence([19, 0]).generate_state(1, dtype=np.uint32)[0])
            expected, _ = direct.reset(seed=worker_seed)
            np.testing.assert_array_equal(obs['lidar'][0], expected['lidar'])
            np.testing.assert_array_equal(obs['state'][0], expected['state'])
        finally:
            direct.close()
            env.close()

    def test_spawn_workers_evaluate_exact_quotas(self):
        import numpy as np
        from baselines.ppo.config import environment
        from baselines.ppo.evaluation import evaluate

        class HoverPolicy:
            def predict(self, obs, deterministic=True):
                return np.zeros((len(obs['state']), 1, 4), dtype=np.float32), None

        cfg = environment('10_sparse')
        cfg['episode_len_sec'] = 0
        records, maps, summary = evaluate(HoverPolicy(), cfg, workers=2, quota=2,
                                          stride=16, subprocess=True)
        self.assertEqual(summary['episodes'], 4)
        self.assertEqual({r['eval_map_seed'] for r in records},
                         {cfg['eval_seed_base'] + offset for offset in (0, 1, 16, 17)})
        self.assertTrue(all(r['episode_length'] == 1 for r in records))
        self.assertEqual(len(maps), 4)


if __name__ == '__main__':
    unittest.main()
