"""Evaluate between completed PPO updates, without resetting training workers."""
from pathlib import Path
import time

from stable_baselines3.common.callbacks import BaseCallback

from .config import write_json, sha256
from .evaluation import evaluate, write_results


class FixedValidation(BaseCallback):
    def __init__(self, config, output, subprocess=True):
        super().__init__()
        self.config, self.output = config, Path(output)
        self.subprocess = subprocess
        self.last_eval = -1
        self.next_eval = config['eval_every']
        self.best_rate = -1.0
        self.started = time.monotonic()

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # SB3 calls this before collecting the next rollout: previous update is complete.
        self.num_timesteps = self.model.num_timesteps
        if self.num_timesteps >= self.next_eval:
            self.validate()

    def _on_training_end(self):
        if self.last_eval != self.num_timesteps:
            self.validate()

    def validate(self):
        self.num_timesteps = self.model.num_timesteps
        cfg = self.config
        records, maps, summary = evaluate(
            self.model, cfg['environment'], cfg['validation_workers'],
            cfg['validation_quota'], cfg['validation_stride'], self.subprocess)
        self.last_eval = self.num_timesteps
        self.next_eval = (self.num_timesteps // cfg['eval_every'] + 1) * cfg['eval_every']
        step = self.num_timesteps
        checkpoint = self.output / f'step_{step:010d}.zip'
        self.model.save(checkpoint)
        summary.update(training_steps=step, elapsed_seconds=time.monotonic() - self.started,
                       checkpoint=checkpoint.name, checkpoint_sha256=sha256(checkpoint), split='validation')
        write_results(self.output / 'validation' / f'step_{step:010d}', records, maps, summary)
        # Ties retain the earliest checkpoint; test scores never enter selection.
        if summary['success_rate'] > self.best_rate:
            self.best_rate = summary['success_rate']
            self.model.save(self.output / 'best_model.zip')
            write_json(self.output / 'best_model.json', summary)
        self.model.save(self.output / 'last_model.zip')
        for name in ('success_rate', 'collision_rate', 'crash_rate', 'timeout_rate'):
            self.logger.record('validation/' + name, summary[name])
        self.logger.dump(step)
        print(f'PPO validation step={step}: success={summary["success_rate"]:.3f}, '
              f'collision={summary["collision_rate"]:.3f}', flush=True)
