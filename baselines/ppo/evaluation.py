"""Fixed quotas and terminal records shared by validation and final testing."""
import csv
import json
import platform
import subprocess
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from cloud.final_test_suite import summarize
from .config import ROOT, sha256, write_json
from .envs import vector_env
from .protocol import EpisodeLedger

PROTOCOL = 'ppo_deterministic_fixed_maps_v1'


def evaluate(model, settings, workers=16, quota=8, stride=16, subprocess=True):
    ledger = EpisodeLedger(workers, quota, stride, int(settings['eval_seed_base']))
    env = vector_env(settings, workers, evaluation=True, quota=quota, stride=stride,
                     subprocess=subprocess)
    try:
        obs = env.reset()
        # Guard against broken termination, while retaining the environment's > time-limit convention.
        limit = quota * (int(settings['episode_len_sec'] * settings['ctrl_freq']) + 3)
        for _ in range(limit):
            actions, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(actions)
            ledger.step(dones, infos)
            if ledger.complete:
                break
    finally:
        env.close()
    records, maps = ledger.finish()
    summary = summarize(records, settings['ctrl_freq'])
    summary.update(evaluation_protocol=PROTOCOL, deterministic=True)
    return records, maps, summary


def provenance():
    packages = {}
    for name in ('stable-baselines3', 'torch', 'numpy', 'gymnasium', 'pybullet'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unknown'
    return dict(package_versions=packages, platform=platform.platform(),
                git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def write_results(directory, records, maps, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # Same sorted JSON row encoding/order as eval_final.py for comparable geometry hashes.
    (directory / 'maps.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in maps))
    (directory / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    for name, rows in [('episodes', records), ('summary', [summary])]:
        if name == 'summary':
            summary['geometry_sha256'] = sha256(directory / 'maps.jsonl')
        with (directory / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(directory / 'summary.json', summary)
