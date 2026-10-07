"""Evaluate a selected checkpoint on a frozen final-test manifest (or materialize maps only)."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
from functools import partial
from pathlib import Path

from final_test_suite import ROOT, validate_manifest, validate_coverage, summarize
from checkpoint_step import resolve_generation


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        if (path / 'agent.pkl').is_file() and (path / 'done').is_file():
            return path
        return resolve_generation(path)
    raise FileNotFoundError(path)


def make_recording_env(config, index, map_dir):
    from dreamerv3.main import make_env
    env = make_env(config, index, eval_mode=True)
    return RecordingEnv(env, map_dir)


class RecordingEnv:
    """Save initial geometry at reset before any dynamic movement."""
    def __init__(self, env, directory):
        self.env, self.directory = env, Path(directory)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, action):
        obs = self.env.step(action)
        if obs['is_first']:
            physics = self.env
            # Unwrap Embodied wrappers and the Gymnasium adapter.
            while not hasattr(type(physics), '_resample_task'):
                physics = physics.env
            row = dict(seed=int(physics.CURRENT_EVAL_SEED),
                       start=physics.INIT_XYZS[0].tolist(), goal=physics.TARGET_POS.tolist(),
                       static_boxes=[list(map(float, x)) for x in physics._obstacle_specs],
                       dynamic_cylinders=[list(map(float, x)) for x in physics._dynamic_obstacle_specs])
            text = json.dumps(row, sort_keys=True)
            path = self.directory / f"{row['seed']}.json"
            if path.exists() and path.read_text() != text:
                raise ValueError('Map reset produced different geometry for the same seed')
            path.write_text(text)
        return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Selected policy run config.yaml')
    parser.add_argument('--checkpoint', help='Agent checkpoint, complete generation, or rolling ckpt directory')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maps-only', action='store_true', help='Generate actual map geometry without loading an agent')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = parser.parse_args()
    if not args.maps_only and not args.checkpoint:
        parser.error('--checkpoint is required unless --maps-only is used')
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text()))
    checkpoint = None if args.maps_only else resolve_checkpoint(args.checkpoint)
    checkpoint_hash = (file_sha256(checkpoint / 'agent.pkl' if checkpoint.is_dir() else checkpoint)
                       if checkpoint else None)

    import elements
    import numpy as np
    from ruamel import yaml
    from dreamerv3.main import make_agent
    from embodied.run.evaluation import evaluate, PROTOCOL
    reader = yaml.YAML(typ='safe')
    saved = reader.load(Path(args.config).read_text())
    config = elements.Config(saved)
    if config.task != 'drone_nav':
        raise ValueError('Expected a drone_nav policy configuration')
    presets = reader.load((ROOT / 'third_party/dreamerv3/dreamerv3/configs.yaml').read_text())
    # Preserve policy architecture/perception, but fix task geometry and dynamics.
    env_changes = {k: v for k, v in presets['defaults']['env']['drone'].items()
                   if k.startswith('dynamic_')}
    for name in manifest['presets']:
        env_changes.update(presets[name]['env.drone'])
    env_changes.update(eval_seed_base=manifest['seed_base'], eval_seed_stride=16,
                       eval_maps_per_env=16, log_image=False, gui=False, use_index=True,
                       randomize_start=True, randomize_goal=True)
    # Reject changes to map-generation/physics settings that would change this suite.
    for key in ('ctrl_freq', 'pyb_freq', 'goal_tolerance', 'speed_limit',
                'collision_distance', 'path_clearance', 'path_cell_size',
                'dynamic_obstacle_clear_radius'):
        env_changes[key] = presets['defaults']['env']['drone'][key]
    config = config.update({'env.drone.' + k: v for k, v in env_changes.items()})
    config = config.update({'logdir': str(args.output.resolve()), 'run.eval_envs': 16,
                            'run.eval_eps': 256, 'jax.platform': 'cuda',
                            'jax.compute_dtype': args.dtype})
    old = saved['env']['drone']
    old_workers = int(saved['run']['eval_envs'])
    validation_seeds = {int(old['eval_seed_base']) + s * int(old['eval_seed_stride']) + w
                        for s in range(int(old['eval_maps_per_env'])) for w in range(old_workers)}
    if validation_seeds.intersection(manifest['map_seeds']):
        raise ValueError('Final test overlaps the supplied run validation seeds')
    args.output.mkdir(parents=True, exist_ok=False)
    map_dir = args.output.resolve() / 'maps'
    map_dir.mkdir()
    config.save(str(args.output / 'config.yaml'))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    factories = [partial(make_recording_env, config, i, str(map_dir)) for i in range(16)]
    if args.maps_only:
        for factory in factories:
            env = factory()
            try:
                for _ in range(16):
                    action = {k: np.zeros(v.shape, v.dtype) for k, v in env.act_space.items()}
                    action['reset'] = np.array(True)
                    env.step(action)
            finally:
                env.close()
        records = None
    else:
        agent = make_agent(config)
        elements.checkpoint.load(str(checkpoint), {'agent': agent.load})
        records, _ = evaluate(agent, factories, 16, parallel=True)
        (args.output / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        validate_coverage(records, manifest)
    maps = [json.loads((map_dir / f'{seed}.json').read_text()) for seed in manifest['map_seeds']]
    if len(list(map_dir.glob('*.json'))) != 256:
        raise ValueError('Expected exactly 256 materialized maps')
    geometry = [(r['start'], r['goal'], r['static_boxes']) for r in maps]
    if len({json.dumps(x) for x in geometry}) != 256:
        raise ValueError('Duplicate map geometry')
    (args.output / 'maps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in maps))
    summary = summarize(records, config.env.drone.ctrl_freq) if records else {'maps': 256}
    summary.update(scenario=manifest['scenario'], manifest_sha256=manifest['sha256'],
                   evaluation_protocol=PROTOCOL, checkpoint=str(checkpoint) if checkpoint else None,
                   checkpoint_sha256=checkpoint_hash,
                   dtype=args.dtype, geometry_sha256=hashlib.sha256(
                       (args.output / 'maps.jsonl').read_bytes()).hexdigest())
    summary['package_versions'] = {}
    for package in ('numpy', 'gymnasium', 'pybullet', 'jax', 'jaxlib', 'elements', 'gym-pybullet-drones'):
        try:
            summary['package_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            summary['package_versions'][package] = 'unknown'
    summary['git_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if records:
        for filename, rows in [('episodes.csv', records), ('summary.csv', [summary])]:
            with (args.output / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
