"""Evaluate an agent-only best checkpoint on the saved run's evaluation maps."""
import argparse
import json
from pathlib import Path
from functools import partial

import elements
import embodied
import ruamel.yaml as yaml
from dreamerv3.main import make_agent, make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = elements.Config(yaml.YAML(typ='safe').load(Path(args.config).read_text()))
    config = config.update({'logdir': str(output), 'jax.platform': 'cuda',
                            'jax.compute_dtype': args.dtype, 'env.drone.log_image': False})
    workers = int(config.run.eval_envs)
    quota = int(config.env.drone.eval_maps_per_env)
    if workers * quota != int(config.run.eval_eps):
        raise ValueError('eval_envs * eval_maps_per_env must equal eval_eps')
    config.save(str(output / 'config.yaml'))
    agent = make_agent(config)
    elements.checkpoint.load(args.checkpoint, {'agent': agent.load})
    records = []

    def on_step(tran, worker):
        if not tran['is_last']:
            return
        record = {'worker': int(worker)}
        for key, value in tran.items():
            if key.startswith('log/') and getattr(value, 'ndim', None) == 0:
                parts = key.split('/')
                name = parts[-1]
                if name != 'video_recorded':
                    record[name] = float(value)
        records.append(record)
        with (output / 'episodes.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')

    driver = embodied.Driver(
        [partial(make_env, config, i, eval_mode=True) for i in range(workers)],
        parallel=True)
    driver.on_step(on_step)
    try:
        driver.reset(agent.init_policy)
        driver(lambda *a: agent.policy(*a, mode='eval'), episodes_per_env=quota)
    finally:
        driver.close()
    if len(records) != workers * quota:
        raise RuntimeError(f'Expected {workers * quota} episodes, got {len(records)}')
    seeds = [int(r['eval_map_seed']) for r in records]
    expected = {int(config.env.drone.eval_seed_base) + i * int(config.env.drone.eval_seed_stride) + w
                for i in range(quota) for w in range(workers)}
    if len(set(seeds)) != len(seeds) or set(seeds) != expected:
        raise RuntimeError('Evaluation map coverage mismatch; inspect episodes.jsonl')
    summary = {'checkpoint': args.checkpoint, 'dtype': args.dtype, 'episodes': len(records)}
    for key in ['success', 'collision', 'crash', 'timeout']:
        summary[key] = sum(r[key] for r in records) / len(records)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
