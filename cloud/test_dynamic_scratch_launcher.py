"""Dependency-free checks that the baseline cannot accidentally finetune."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name('run_dynamic_scratch.sh')


class ScratchLauncherTest(unittest.TestCase):
    def launch(self, root, **overrides):
        env = dict(os.environ)
        for key in ('MODE', 'LOGDIR', 'BATCH_SIZE', 'RESUME_BATCH_SIZE'):
            env.pop(key, None)
        env.update(TRAIN_PYTHON=sys.executable, DRY_RUN='1', LOG_ROOT=root,
                   SEED='2', STEPS='3000000', INIT_CHECKPOINT='/must/not/load.ckpt')
        env.update(overrides)
        return subprocess.run(['bash', str(SCRIPT)], env=env,
                              text=True, capture_output=True, timeout=15)

    def test_random_initialization_and_separate_directory(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.launch(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = shlex.split(result.stdout)
            self.assertEqual(args[args.index('--run.from_checkpoint') + 1], '')
            self.assertNotIn('/must/not/load.ckpt', args)
            self.assertEqual(args[args.index('--batch_size') + 1], '32')
            self.assertEqual(args[args.index('--logdir') + 1], str(Path(root) / 'seed_2'))
            self.assertIn('drone_dynamic_20_sparse', args)
            self.assertIn('drone_perc_cnn_hi', args)

    def test_explicit_batch_and_budget(self):
        with tempfile.TemporaryDirectory() as root:
            result = self.launch(root, BATCH_SIZE='16', STEPS='5600000')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = shlex.split(result.stdout)
            self.assertEqual(args[args.index('--batch_size') + 1], '16')
            self.assertEqual(args[args.index('--run.steps') + 1], '5600000')

    def test_existing_run_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / 'seed_2'
            run.mkdir()
            (run / 'config.yaml').write_text('existing run')
            result = self.launch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('empty LOGDIR', result.stderr)
            self.assertEqual((run / 'config.yaml').read_text(), 'existing run')

    def test_finetune_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertNotEqual(self.launch(root, MODE='finetune').returncode, 0)


if __name__ == '__main__':
    unittest.main()
