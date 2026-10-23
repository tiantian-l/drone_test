"""Resolve the same environment presets as the Dreamer launchers."""
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
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
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


def environment(scenario, source_config=None):
    """Use a resolved Dreamer config, or the launcher's ordered presets."""
    if source_config:
        saved = yaml.safe_load(Path(source_config).read_text())
        if saved.get('task') != 'drone_nav':
            raise ValueError('Source config must be a resolved drone_nav Dreamer config')
        result = copy.deepcopy(saved['env']['drone'])
        expected = environment(scenario)
        # Reject accidentally labeling a different training task as this scenario.
        for key in ('map_range', 'n_obstacles', 'start_sample_range', 'goal_sample_range',
                    'episode_len_sec', 'dynamic_obstacles_enabled'):
            if result.get(key, False) != expected[key]:
                raise ValueError(f'Source config does not match {scenario}: {key}')
    else:
        data = presets()
        result = copy.deepcopy(data['defaults']['env']['drone'])
        for name in ['drone_nav', 'drone_perc_cnn_hi', *scene_presets(scenario), 'drone_reward_v2']:
            result = merge(result, data[name].get('env.drone', {}))
    if result.get('perception') != 'split':
        raise ValueError('CNN-PPO requires split state/LiDAR observations')
    result.update(gui=False, log_image=False)
    return result


def test_environment(training, manifest):
    """Mirror eval_final.py: retain perception/reward; enforce test task/physics."""
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
