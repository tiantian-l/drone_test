"""Train CNN-PPO, defaulting to a 10_sparse feasibility run."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/gym-pybullet-drones'))

import yaml
from baselines.ppo.config import environment, SCENARIOS, write_json, sha256
from cloud.final_test_suite import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ppo.yaml')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--source-config', type=Path, help='Resolved Dreamer config for exact environment matching')
    parser.add_argument('--output', type=Path, required=True, help='New run directory (never overwritten)')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int, help='New steps; with --resume, additional steps')
    parser.add_argument('--envs', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--resume', type=Path, help='Checkpoint from a run with adjacent config.json')
    parser.add_argument('--serial', action='store_true', help='Use in-process workers for debugging')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    resume_cfg = None
    if args.resume:
        import json
        resume_cfg = json.loads((args.resume.resolve().parent / 'config.json').read_text())
        if resume_cfg['sources'] != source_hashes():
            raise ValueError('Environment sources changed since the resumed run')
        cfg = copy.deepcopy(resume_cfg)
        if args.source_config or (args.scenario and args.scenario != cfg['scenario']):
            raise ValueError('Cannot change the task when resuming a policy')
    for field, value in [('scenario', args.scenario), ('seed', args.seed),
                         ('total_timesteps', args.steps), ('n_envs', args.envs),
                         ('device', args.device), ('eval_every', args.eval_every)]:
        if value is not None:
            cfg[field] = value
    for field in ('total_timesteps', 'n_envs', 'eval_every', 'validation_workers',
                  'validation_quota', 'validation_stride', 'torch_threads'):
        if not isinstance(cfg[field], int) or cfg[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if cfg['validation_stride'] < cfg['validation_workers']:
        raise ValueError('Validation map streams must not overlap')
    rollout = cfg['n_envs'] * cfg['ppo']['n_steps']
    batch = cfg['ppo']['batch_size']
    if batch < 2 or rollout < batch or rollout % batch:
        raise ValueError('Rollout size must be a multiple of batch_size, with batch_size >= 2')
    if not resume_cfg:
        cfg['environment'] = environment(cfg['scenario'], args.source_config)
    cfg['sources'] = source_hashes()
    cfg['source_config'] = str(args.source_config.resolve()) if args.source_config else cfg.get('source_config')
    cfg['resume_checkpoint'] = str(args.resume.resolve()) if args.resume else None
    cfg['resume_checkpoint_sha256'] = sha256(args.resume) if args.resume else None
    cfg['preprocessing'] = 'state_symlog_lidar_identity_v1'
    cfg['selection'] = 'maximum_validation_success_rate_keep_earliest_tie'
    cfg['validation_maps'] = cfg['validation_workers'] * cfg['validation_quota']
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    # Import heavy libraries only after config validation/dry-run.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from baselines.ppo.envs import vector_env
    from baselines.ppo.features import NavigationFeatures
    from baselines.ppo.training import FixedValidation
    from baselines.ppo.evaluation import provenance

    torch.set_num_threads(cfg['torch_threads'])
    set_random_seed(cfg['seed'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', cfg)
    write_json(args.output / 'provenance.json', provenance())
    env = vector_env(cfg['environment'], cfg['n_envs'], seed=cfg['seed'], subprocess=not args.serial)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=cfg['device'])
            model.set_random_seed(cfg['seed'])
        else:
            model = PPO('MultiInputPolicy', env, seed=cfg['seed'], device=cfg['device'],
                        policy_kwargs=dict(features_extractor_class=NavigationFeatures,
                                           normalize_images=False,
                                           net_arch=dict(pi=[256, 256], vf=[256, 256])),
                        verbose=1, **cfg['ppo'])
        model.set_logger(configure(str(args.output / 'logs'), ['stdout', 'csv', 'tensorboard']))
        write_json(args.output / 'model.json', dict(
            parameters=sum(p.numel() for p in model.policy.parameters()),
            action_shape=list(env.action_space.shape), device=str(model.device)))
        start = model.num_timesteps
        callback = FixedValidation(cfg, args.output, subprocess=not args.serial)
        # A resumed run selects its own best checkpoint, including the initial resumed policy.
        if args.resume:
            callback.next_eval = start
        model.learn(total_timesteps=cfg['total_timesteps'], callback=callback,
                    reset_num_timesteps=not bool(args.resume))
        model.save(args.output / 'final_model.zip')
        write_json(args.output / 'completed.json', dict(
            initial_steps=start, actual_steps=model.num_timesteps,
            requested_additional_steps=cfg['total_timesteps'],
            note='Full rollouts may exceed the requested step budget.'))
    finally:
        env.close()


if __name__ == '__main__':
    main()
