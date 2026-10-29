"""Environment settings and evaluation bookkeeping for the planner baseline.

Self-contained copy of the preset resolution used by the Dreamer launchers
(and by the APF/PPO baselines on their own branches), so that this branch does
not depend on baseline code from other branches.
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ROOT / 'third_party/dreamerv3/dreamerv3/configs.yaml'
SCENARIOS = ('10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic')
FRAMEWORK_KEYS = {'use_index', 'use_seed', 'use_logdir', 'log_image', 'video_every',
                  'eval_seed_base', 'eval_maps_per_env', 'eval_seed_stride'}
PHYSICS_KEYS = ('ctrl_freq', 'pyb_freq', 'goal_tolerance', 'speed_limit',
                'collision_distance', 'path_clearance', 'path_cell_size',
                'dynamic_obstacle_clear_radius')


def merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = (merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else copy.deepcopy(value))
    return result


def presets():
    return yaml.safe_load(PRESETS.read_text())


def scene_presets(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    names = ['drone_static_' + ('20_sparse' if scenario == '20_sparse_dynamic' else scenario)]
    if scenario == '20_sparse_dynamic':
        names.append('drone_dynamic_20_sparse')
    return names


def environment(scenario):
    """Launcher preset order: defaults, drone_nav, perception, scene, reward."""
    data = presets()
    result = copy.deepcopy(data['defaults']['env']['drone'])
    for name in ['drone_nav', 'drone_perc_cnn_hi', *scene_presets(scenario), 'drone_reward_v2']:
        result = merge(result, data[name].get('env.drone', {}))
    result.update(gui=False, log_image=False)
    return result


def test_environment(training, manifest):
    """Mirror eval_final.py: enforce the test task and physics of the manifest."""
    data = presets()
    changes = {k: v for k, v in data['defaults']['env']['drone'].items() if k.startswith('dynamic_')}
    for name in manifest['presets']:
        changes.update(data[name]['env.drone'])
    changes.update({k: data['defaults']['env']['drone'][k] for k in PHYSICS_KEYS})
    changes.update(gui=False, log_image=False, randomize_start=True, randomize_goal=True,
                   eval_seed_base=manifest['seed_base'], eval_seed_stride=16, eval_maps_per_env=16)
    return merge(training, changes)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class EpisodeLedger:
    """Checks that each logical worker visits its assigned maps in order."""

    def __init__(self, workers, quota, stride, seed_base):
        if min(workers, quota) < 1 or stride < workers:
            raise ValueError('Positive quotas and stride >= workers are required')
        self.workers, self.quota, self.stride, self.seed_base = workers, quota, stride, seed_base
        self.counts = [0] * workers
        self.lengths = [0] * workers
        self.records, self.maps = [], []

    def step(self, dones, infos):
        for worker, done in enumerate(dones):
            self.lengths[worker] += 1
            if not done:
                continue
            if self.counts[worker] < self.quota:
                info = infos[worker]
                slot = self.counts[worker]
                expected = self.seed_base + slot * self.stride + worker
                if int(info['eval_map_seed']) != expected or int(info['eval_map_slot']) != slot:
                    raise ValueError('Unexpected evaluation map order')
                if int(info['initial_geometry']['seed']) != expected:
                    raise ValueError('Terminal geometry belongs to a different episode')
                row = dict(worker=worker, episode_index_in_cycle=slot, eval_map_slot=slot,
                           eval_map_seed=expected, episode_length=self.lengths[worker])
                for name in ('success', 'crash', 'collision', 'timeout'):
                    row[name] = bool(info['is_' + name])
                for name in ('static_collision', 'dynamic_collision', 'dynamic_count',
                             'final_distance', 'min_distance', 'min_lidar_dist'):
                    row[name] = float(info[name])
                self.records.append(row)
                self.maps.append(info['initial_geometry'])
                self.counts[worker] += 1
            self.lengths[worker] = 0

    def finish(self):
        if any(count != self.quota for count in self.counts):
            raise RuntimeError(f'Incomplete evaluation: {self.counts}')
        records = sorted(self.records, key=lambda r: (r['worker'], r['eval_map_slot']))
        maps = sorted(self.maps, key=lambda r: r['seed'])
        if len({r['seed'] for r in maps}) != self.workers * self.quota:
            raise ValueError('Duplicate evaluation seed')
        return records, maps
