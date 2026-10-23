"""Evaluate a saved PPO policy on one frozen final-test manifest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/gym-pybullet-drones'))

from baselines.ppo.config import test_environment, sha256, write_json
from cloud.final_test_suite import validate_manifest, validate_coverage, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Defaults to config.json next to checkpoint')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--serial', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads((args.config or args.checkpoint.parent / 'config.json').read_text())
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    if cfg['sources'] != source_hashes():
        raise ValueError('Training environment sources differ from the frozen test sources')
    validation_seeds = {int(cfg['environment']['eval_seed_base']) + slot * cfg['validation_stride'] + worker
                        for worker in range(cfg['validation_workers']) for slot in range(cfg['validation_quota'])}
    if validation_seeds.intersection(manifest['map_seeds']):
        raise ValueError('Validation/test map overlap')
    settings = test_environment(cfg['environment'], manifest)

    import torch
    from stable_baselines3 import PPO
    from baselines.ppo.evaluation import evaluate, write_results, provenance
    torch.set_num_threads(cfg['torch_threads'])
    model = PPO.load(args.checkpoint, device=args.device)
    records, maps, summary = evaluate(model, settings, workers=16, quota=16, stride=16,
                                     subprocess=not args.serial)
    validate_coverage(records, manifest)
    if [r['seed'] for r in maps] != manifest['map_seeds']:
        raise ValueError('Map geometry coverage mismatch')
    geometry = [(r['start'], r['goal'], r['static_boxes']) for r in maps]
    if len({json.dumps(g) for g in geometry}) != 256:
        raise ValueError('Duplicate initial geometry')
    summary.update(provenance(), scenario=manifest['scenario'], training_scenario=cfg['scenario'],
                   training_seed=cfg['seed'], training_steps=model.num_timesteps,
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                   manifest_sha256=manifest['sha256'], split='test', device=str(model.device))
    write_results(args.output, records, maps, summary)
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'config.json', dict(training=cfg, evaluation_environment=settings))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
