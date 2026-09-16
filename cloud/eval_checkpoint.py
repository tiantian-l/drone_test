"""Evaluate an agent-only best checkpoint on the saved run's evaluation maps."""
import argparse
import json
from pathlib import Path
from functools import partial

import elements
import embodied
import ruamel.yaml as yaml
from dreamerv3.main import make_agent, make_env
from embodied.run.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--eval-maps', type=int, default=None,
                        help='Override saved evaluation size; otherwise preserve it')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = elements.Config(yaml.YAML(typ='safe').load(Path(args.config).read_text()))
    config = config.update({'logdir': str(output), 'jax.platform': 'cuda',
                            'jax.compute_dtype': args.dtype, 'env.drone.log_image': False})
    workers = int(config.run.eval_envs)
    if args.eval_maps is not None:
        if args.eval_maps <= 0 or args.eval_maps % workers:
            raise ValueError('eval-maps must be positive and divisible by eval_envs')
        config = config.update({'run.eval_eps': args.eval_maps,
                                'env.drone.eval_maps_per_env': args.eval_maps // workers})
    quota = int(config.env.drone.eval_maps_per_env)
    if workers * quota != int(config.run.eval_eps):
        raise ValueError('eval_envs * eval_maps_per_env must equal eval_eps')
    config.save(str(output / 'config.yaml'))
    agent = make_agent(config)
    elements.checkpoint.load(args.checkpoint, {'agent': agent.load})
    records, summary = evaluate(agent,
        [partial(make_env, config, i, eval_mode=True) for i in range(workers)],
        quota, parallel=True)
    (output / 'episodes.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in records))
    if len(records) != workers * quota:
        raise RuntimeError(f'Expected {workers * quota} episodes, got {len(records)}')
    seeds = [int(r['eval_map_seed']) for r in records]
    expected = {int(config.env.drone.eval_seed_base) + i * int(config.env.drone.eval_seed_stride) + w
                for i in range(quota) for w in range(workers)}
    if len(set(seeds)) != len(seeds) or set(seeds) != expected:
        raise RuntimeError('Evaluation map coverage mismatch; inspect episodes.jsonl')
    summary.update(checkpoint=args.checkpoint, dtype=args.dtype)
    for key in ['success', 'collision', 'crash', 'timeout']:
        summary[key] = sum(r[key] for r in records) / len(records)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
