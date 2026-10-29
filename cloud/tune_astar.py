"""Per-scenario validation tuning of the map-based planner baseline (Map-A*).

Same protocol and budget as the APF tuning on branch feature/apf
(cloud/tune_apf.py there), so that the two baselines are tuned equally:
  1. Candidates: implementation defaults (incumbent) plus 31 Latin-hypercube
     samples from SEARCH_SPACE (sampler seed SAMPLER_SEED); the same samples are
     used for every scenario.
  2. Successive halving on the scenario's own 128 validation seeds
     (eval_seed_base .. eval_seed_base+127, the Dreamer validation pool):
        rung 0: all 32 candidates on maps   0-31
        rung 1: best 8           on maps  32-63   (64 maps total)
        rung 2: best 3           on maps  64-127  (128 maps total)
     The incumbent is additionally completed on all 128 maps as a reference.
  3. Ranking (cumulative over all maps a candidate has seen):
        more successes > fewer collisions+crashes > shorter mean success time.
     The selected configuration is the rung-2 winner.
Test seeds (>= 1,000,000) are never touched; overlap is rejected.

Each episode runs in a reused environment whose evaluation seed is set
explicitly; reset() reseeds the RNG and clears the PID state, so an episode
depends only on (scenario, parameters, map seed). Results are appended to
episodes.jsonl and the run resumes from it after interruption.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/gym-pybullet-drones'))

from baselines.astar.policy import AStarConfig, AStarPolicy  # noqa: E402
from baselines.astar.settings import environment, FRAMEWORK_KEYS  # noqa: E402

SCENARIOS = ('10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic')
# Incumbent = implementation defaults. (scale, low, high); cruise_speed <= 2 m/s limit.
INCUMBENT = {k: v for k, v in AStarConfig().__dict__.items()}
SEARCH_SPACE = {
    'cruise_speed': ('lin', 0.8, 2.0),
    'inflation': ('lin', 0.35, 0.6),
    'clearance_weight': ('log', 0.5, 16.0),
    'lookahead': ('lin', 0.5, 2.0),
    'slow_clearance': ('lin', 0.4, 1.5),
    'safety_radius': ('lin', 0.4, 0.8),
}
SAMPLER_SEED = 20261026
N_RANDOM = 31
RUNGS = ((32, 32), (8, 64), (3, 128))  # (candidates kept, cumulative maps)
VALIDATION_MAPS = 128


def candidates():
    from scipy.stats import qmc
    config_type, incumbent, space = AStarConfig, INCUMBENT, SEARCH_SPACE
    unit = qmc.LatinHypercube(d=len(space), seed=SAMPLER_SEED).random(N_RANDOM)
    result = [dict(id='c00_incumbent', params=dict(incumbent))]
    for i, row in enumerate(unit, start=1):
        params = dict(incumbent)
        for u, (key, (scale, lo, hi)) in zip(row, space.items()):
            value = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if scale == 'log' else lo + u * (hi - lo)
            params[key] = round(float(value), 4)
        config_type(**params)  # validates
        result.append(dict(id=f'c{i:02d}', params=params))
    return result


def settings_for(scenario):
    settings = environment(scenario)
    base = int(settings['eval_seed_base'])
    if base + VALIDATION_MAPS > 1_000_000:
        raise ValueError('Validation seeds overlap the test range')
    return settings


_ENVS = {}


def _env(scenario):
    if scenario not in _ENVS:
        from drone_nav.envs.nav_aviary import NavigationAviary
        settings = settings_for(scenario)
        kwargs = {k: v for k, v in settings.items() if k not in FRAMEWORK_KEYS}
        kwargs.update(gui=False, log_video=False, render_mode='2d', eval_mode=True,
                      eval_seed=int(settings['eval_seed_base']), eval_maps_per_env=1,
                      eval_seed_stride=1)
        _ENVS[scenario] = (NavigationAviary(**kwargs), settings)
    return _ENVS[scenario]


def run_episode(task):
    scenario, cid, params, seed, task_info = task
    env, settings = _env(scenario)
    policy = AStarPolicy.from_environment(settings, AStarConfig(**params), task_info=task_info)
    env.EVAL_SEED = int(seed)
    env._eval_episode_counter = 0
    start = time.monotonic()
    obs, _ = env.reset()
    # Tuning protocol: start every validation episode from a cleared PID state,
    # so an episode depends only on (parameters, map seed) regardless of which
    # episode the worker ran before. No-op if the environment already does it.
    for controller in env.ctrl:
        controller.reset()
    policy.reset()
    policy.set_task(env.TARGET_POS)  # declared task information (ignored with task_info='none')
    assert int(env.CURRENT_EVAL_SEED) == int(seed)
    start_dist = float(np.linalg.norm(obs['state'][:3]))
    prev = np.asarray(obs['state'][:3], dtype=float)
    path = 0.0
    limit = int(env.EPISODE_LEN_SEC * env.CTRL_FREQ) + 3
    for step in range(limit):
        action = policy.act(obs)
        obs, _, terminated, truncated, info = env.step(action.reshape(1, 4))
        rel = np.asarray(obs['state'][:3], dtype=float)
        path += float(np.linalg.norm(rel - prev))  # goal is fixed: |d rel| = |d pos|
        prev = rel
        if terminated or truncated:
            break
    else:
        raise RuntimeError('Environment did not terminate')
    success = bool(info['is_success'])
    return dict(scenario=scenario, candidate=cid, seed=int(seed), steps=step + 1,
                success=success, collision=bool(info['is_collision']), crash=bool(info['is_crash']),
                timeout=bool(info['is_timeout']), static_collision=bool(info['static_collision']),
                dynamic_collision=bool(info['dynamic_collision']),
                final_distance=float(info['final_distance']), start_distance=start_dist,
                path_length=path,
                spl=(start_dist / max(path, start_dist)) if success else 0.0,
                time_s=(step + 1) / float(env.CTRL_FREQ), wall_s=time.monotonic() - start)


def score(rows):
    n = len(rows)
    succ = sum(r['success'] for r in rows)
    fail = sum(r['collision'] or r['crash'] for r in rows)
    times = [r['time_s'] for r in rows if r['success']]
    mean_t = sum(times) / len(times) if times else float('inf')
    return (succ / n, -fail / n, -mean_t)


def summary(rows):
    n = len(rows)
    out = dict(episodes=n)
    for k in ('success', 'collision', 'crash', 'timeout', 'static_collision', 'dynamic_collision'):
        out[k + '_rate'] = sum(r[k] for r in rows) / n
    out['success_count'] = sum(r['success'] for r in rows)
    z = 1.959963984540054
    p = out['success_rate']
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    out['success_ci95'] = [max(0.0, c - h), min(1.0, c + h)]
    succ = [r for r in rows if r['success']]
    out['success_time_mean'] = sum(r['time_s'] for r in succ) / len(succ) if succ else None
    out['success_path_mean'] = sum(r['path_length'] for r in succ) / len(succ) if succ else None
    out['spl'] = sum(r['spl'] for r in rows) / n
    return out


class Store:
    def __init__(self, path):
        self.path = path
        self.rows = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.rows[(r['scenario'], r['candidate'], r['seed'])] = r

    def add(self, r):
        self.rows[(r['scenario'], r['candidate'], r['seed'])] = r
        with self.path.open('a') as f:
            f.write(json.dumps(r) + '\n')

    def get(self, scenario, cid, seeds):
        return [self.rows[(scenario, cid, s)] for s in seeds]


TASK_INFO = 'region'


def evaluate(pool, store, scenario, cands, seeds, log):
    tasks = [(scenario, c['id'], c['params'], s, TASK_INFO) for c in cands for s in seeds
             if (scenario, c['id'], s) not in store.rows]
    # Longest-first is unknown; interleave candidates for balanced load.
    tasks.sort(key=lambda t: (t[3], t[1]))
    t0, done = time.monotonic(), 0
    for r in pool.imap_unordered(run_episode, tasks, chunksize=1):
        store.add(r)
        done += 1
        if done % 25 == 0 or done == len(tasks):
            log(f'  {scenario}: {done}/{len(tasks)} episodes, {time.monotonic() - t0:.0f}s')


def tune(scenario, pool, store, cands, out, log):
    base = int(settings_for(scenario)['eval_seed_base'])
    seeds_all = list(range(base, base + VALIDATION_MAPS))
    alive = list(cands)
    history = []
    for rung, (keep, maps) in enumerate(RUNGS):
        if rung > 0:
            alive = ranked[:keep]
        seeds = seeds_all[:maps]
        log(f'{scenario} rung {rung}: {len(alive)} candidates x {maps} maps')
        evaluate(pool, store, scenario, alive, seeds, log)
        ranked = sorted(alive, key=lambda c: score(store.get(scenario, c['id'], seeds)), reverse=True)
        history.append(dict(rung=rung, maps=maps, ranking=[
            dict(id=c['id'], **summary(store.get(scenario, c['id'], seeds))) for c in ranked]))
    best = ranked[0]
    incumbent = cands[0]
    evaluate(pool, store, scenario, [incumbent], seeds_all, log)
    result = dict(scenario=scenario, seeds=[seeds_all[0], seeds_all[-1]],
                  selected=best['id'], selected_params=best['params'],
                  selected_summary=summary(store.get(scenario, best['id'], seeds_all)),
                  incumbent_summary=summary(store.get(scenario, incumbent['id'], seeds_all)),
                  rungs=history)
    (out / f'{scenario}_result.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / f'{scenario}_frozen_params.json').write_text(json.dumps(best['params'], indent=2) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--task-info', choices=('region', 'none'), default='region',
                    help="'region': goal world position + flight region; 'none': observation only")
    ap.add_argument('--only-reproduce', type=str, default=None,
                    help='scenario:n — run the incumbent on n validation maps and stop')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    global TASK_INFO
    TASK_INFO = args.task_info
    cands = candidates()
    plan = dict(algorithm='astar', task_info=args.task_info, search_space=SEARCH_SPACE,
                fixed={k: INCUMBENT[k] for k in INCUMBENT if k not in SEARCH_SPACE},
                sampler='scipy LatinHypercube', sampler_seed=SAMPLER_SEED, rungs=RUNGS,
                ranking='success desc, collision+crash asc, mean success time asc',
                candidates=cands,
                policy_sha256=hashlib.sha256((ROOT / 'baselines/astar/policy.py').read_bytes()).hexdigest(),
                env_sha256=hashlib.sha256((ROOT / 'drone_nav/envs/nav_aviary.py').read_bytes()).hexdigest())
    plan_path = args.output / 'plan.json'
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old['candidates'] != json.loads(json.dumps(cands)) or old.get('task_info', 'region') != args.task_info:
            raise ValueError('Existing plan differs; refusing to mix runs')
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    store = Store(args.output / 'episodes.jsonl')
    logf = (args.output / 'progress.log').open('a')

    def log(msg):
        line = time.strftime('%H:%M:%S ') + msg
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    with mp.get_context('fork').Pool(args.workers) as pool:
        if args.only_reproduce:
            scenario, n = args.only_reproduce.split(':')
            base = int(settings_for(scenario)['eval_seed_base'])
            evaluate(pool, store, scenario, cands[:1], list(range(base, base + int(n))), log)
            return
        for scenario in args.scenarios:
            r = tune(scenario, pool, store, cands, args.output, log)
            log(f"{scenario}: selected {r['selected']} success {r['selected_summary']['success_rate']:.3f} "
                f"vs incumbent {r['incumbent_summary']['success_rate']:.3f}")


if __name__ == '__main__':
    main()
