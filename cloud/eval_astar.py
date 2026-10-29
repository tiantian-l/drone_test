"""CPU-only validation/frozen-test evaluation of the map-based planner (Map-A*).

Mirrors cloud/eval_apf.py on branch feature/apf: same manifests, worker/slot
schedule, ledger checks and summary format. The planner additionally receives
the declared task information (goal position in the world frame; the flight
region comes from the environment settings) once per episode."""
import argparse
import csv
import subprocess
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/gym-pybullet-drones'))

from baselines.astar.policy import AStarConfig, AStarPolicy
from baselines.astar.settings import environment, test_environment, FRAMEWORK_KEYS, write_json, EpisodeLedger
from cloud.final_test_suite import summarize, validate_manifest, validate_coverage, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic'), default='10_sparse')
    parser.add_argument('--params', type=Path, help='JSON object containing AStarConfig fields')
    parser.add_argument('--task-info', choices=('region', 'none'), default='region',
                        help="'region': goal world position + flight region; 'none': observation only")
    parser.add_argument('--manifest', type=Path, help='Frozen final-test manifest; runs all 256 maps')
    parser.add_argument('--episodes', type=int, default=4, help='Validation maps (1..128); ignored for final test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gif', action='store_true', help='Save a top-down animation of the first map')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.episodes <= 128:
        parser.error('--episodes must be in 1..128')
    manifest = validate_manifest(json.loads(args.manifest.read_text())) if args.manifest else None
    scenario = manifest['scenario'] if manifest else args.scenario
    settings = environment(scenario)
    if manifest:
        settings = test_environment(settings, manifest)
    params = AStarConfig(**json.loads(args.params.read_text())) if args.params else AStarConfig()
    policy = AStarPolicy.from_environment(settings, params, task_info=args.task_info)
    algorithm = 'Map-A*' if args.task_info == 'region' else 'Map-A* (no region)'
    from drone_nav.envs.nav_aviary import NavigationAviary
    kwargs = {k: v for k, v in settings.items() if k not in FRAMEWORK_KEYS}
    unknown = set(kwargs) - set(inspect.signature(NavigationAviary.__init__).parameters)
    if unknown:
        raise ValueError(f'Unknown environment options: {unknown}')
    kwargs.update(gui=False, log_video=False, render_mode='2d')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', dict(scenario=scenario, environment=settings,
               planner=asdict(params), algorithm=algorithm, sources=source_hashes(),
               task_info=args.task_info, privileged_task_info=list(policy.PRIVILEGED_TASK_INFO),
               policy_sha256=hashlib.sha256((ROOT / 'baselines/astar/policy.py').read_bytes()).hexdigest(),
               git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               evaluation_schedule=dict(workers=16 if manifest else 1, quota=16 if manifest else args.episodes, stride=16 if manifest else 1),
               parameter_status='user_supplied' if args.params else 'untuned_defaults'))
    if manifest:
        write_json(args.output / 'manifest.json', manifest)
    # Serial execution preserves the same logical worker/slot/seed assignments.
    workers, quota, stride = (16, 16, 16) if manifest else (1, args.episodes, 1)
    base = manifest['seed_base'] if manifest else int(settings['eval_seed_base'])
    records, maps, frames = [], [], []
    trace = (args.output / 'trajectories.jsonl').open('w')
    try:
        for worker in range(workers):
            ledger = EpisodeLedger(1, quota, stride, base + worker)
            env = NavigationAviary(**dict(kwargs, eval_mode=True, eval_seed=base + worker,
                                         eval_maps_per_env=quota, eval_seed_stride=stride))
            try:
                for slot in range(quota):
                    obs, _ = env.reset()
                    policy.reset()
                    policy.set_task(env.TARGET_POS)  # ignored with --task-info none
                    geometry = dict(seed=int(env.CURRENT_EVAL_SEED), start=env.INIT_XYZS[0].tolist(),
                                    goal=env.TARGET_POS.tolist(),
                                    static_boxes=[list(map(float, x)) for x in env._obstacle_specs],
                                    dynamic_cylinders=[list(map(float, x)) for x in env._dynamic_obstacle_specs])
                    # Geometry is recorded for auditing only; never passed to policy.
                    limit = int(env.EPISODE_LEN_SEC * env.CTRL_FREQ) + 3
                    for step in range(limit):
                        action = policy.act(obs)
                        state = obs['state'] if isinstance(obs, dict) else obs[:policy.state_size]
                        trace.write(json.dumps(dict(seed=geometry['seed'], step=step,
                            position=(env.TARGET_POS - state[:3]).tolist(),
                            velocity=state[3:6].tolist(), action=action.tolist(),
                            diagnostics=getattr(policy, 'diagnostics', None))) + '\n')
                        if args.gif and worker == 0 and slot == 0 and step % 3 == 0:
                            from PIL import Image
                            frames.append(Image.fromarray(env.render_frame()))
                        obs, _, terminated, truncated, info = env.step(action.reshape(1, 4))
                        done = terminated or truncated
                        if done:
                            info = dict(info, initial_geometry=geometry)
                        ledger.step([done], [info])
                        if done:
                            print(json.dumps(dict(seed=geometry['seed'], success=info['is_success'],
                                collision=info['is_collision'], crash=info['is_crash'], timeout=info['is_timeout'], steps=step + 1)), flush=True)
                            break
                    else:
                        raise RuntimeError('Environment did not terminate within its time limit')
                rows, geometries = ledger.finish()
                for row in rows:
                    row['worker'] = worker
                records.extend(rows)
                maps.extend(geometries)
            finally:
                env.close()
    finally:
        trace.close()
    maps.sort(key=lambda row: row['seed'])
    if manifest:
        validate_coverage(records, manifest)
        if [r['seed'] for r in maps] != manifest['map_seeds']:
            raise ValueError('Map geometry coverage mismatch')
        geometry_keys = [json.dumps((r['start'], r['goal'], r['static_boxes'])) for r in maps]
        if len(set(geometry_keys)) != len(maps):
            raise ValueError('Duplicate initial geometry')
    for name, rows in [('episodes', records), ('maps', maps)]:
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, sort_keys=name == 'maps') + '\n' for r in rows))
    summary = summarize(records, settings['ctrl_freq'])
    summary.update(scenario=scenario, algorithm=algorithm, deterministic=True,
                   split='test' if manifest else 'validation',
                   evaluation_protocol='map_astar_pid_reset_v1', task_info=args.task_info,
                   geometry_sha256=hashlib.sha256((args.output / 'maps.jsonl').read_bytes()).hexdigest())
    write_json(args.output / 'summary.json', summary)
    for name, rows in [('episodes', records), ('summary', [summary])]:
        with (args.output / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if frames:
        frames[0].save(args.output / 'first_episode.gif', save_all=True, append_images=frames[1:],
                       duration=round(3000 / settings['ctrl_freq']), loop=0)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
