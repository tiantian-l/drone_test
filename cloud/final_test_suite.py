"""Versioned held-out map recipes; generation and validation need only stdlib."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic')
SOURCES = ('third_party/dreamerv3/dreamerv3/configs.yaml',
           'drone_nav/envs/nav_aviary.py', 'drone_nav/moving_geometry.py',
           'drone_nav/from_gymnasium.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def build_manifest(scenario):
    index = SCENARIOS.index(scenario)
    base = 1000000 + index * 10000
    static = '20_sparse' if scenario == '20_sparse_dynamic' else scenario
    presets = ['drone_static_' + static]
    if scenario == '20_sparse_dynamic':
        presets.append('drone_dynamic_20_sparse')
    value = dict(schema='drone_final_test_v1', scenario=scenario, maps=256,
                 workers=16, quota=16, seed_base=base, seed_stride=16,
                 map_seeds=list(range(base, base + 256)), presets=presets,
                 sources=source_hashes())
    value['sha256'] = digest(value)
    return value


def validate_manifest(value, check_sources=True):
    content = {k: v for k, v in value.items() if k != 'sha256'}
    if value.get('sha256') != digest(content):
        raise ValueError('Test manifest checksum mismatch')
    if value.get('scenario') not in SCENARIOS:
        raise ValueError('Unknown test scenario')
    expected = build_manifest(value['scenario'])
    keys = set(expected) - {'sha256', 'sources'}
    if any(value.get(k) != expected[k] for k in keys):
        raise ValueError('Unsupported test protocol or map schedule')
    if check_sources and value['sources'] != source_hashes():
        raise ValueError('Map generator/configuration changed since manifest generation; use the frozen code version')
    return value


def validate_coverage(records, manifest):
    expected = {(w, s, manifest['seed_base'] + s * 16 + w)
                for w in range(16) for s in range(16)}
    actual = [(int(r['worker']), int(r['eval_map_slot']), int(r['eval_map_seed']))
              for r in records]
    if len(actual) != 256 or set(actual) != expected:
        raise ValueError('Final test coverage mismatch: expected 256 distinct worker/slot/seed tuples')


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    den = 1 + z*z/count
    center = (p + z*z/(2*count)) / den
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / den
    return [max(0, center-half), min(1, center+half)]


def summarize(records, ctrl_freq):
    result = {'episodes': len(records)}
    for name in ('success', 'collision', 'crash', 'timeout', 'static_collision', 'dynamic_collision'):
        if not all(name in r for r in records):
            continue
        count = sum(bool(r[name]) for r in records)
        result[name + '_count'] = count
        result[name + '_rate'] = count / len(records)
        result[name + '_ci95'] = wilson(count, len(records))
    for name in ('final_distance', 'min_distance', 'episode_length'):
        values = [float(r[name]) for r in records if name in r and math.isfinite(float(r[name]))]
        result[name + '_mean'] = sum(values) / len(values) if values else None
    times = [r['episode_length'] / ctrl_freq for r in records if r['success']]
    result['successful_time_seconds_mean'] = sum(times) / len(times) if times else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for scenario in SCENARIOS:
        value = build_manifest(scenario)
        validate_manifest(value)
        (args.output / f'{scenario}.json').write_text(json.dumps(value, indent=2) + '\n')
    print(f'Generated five fixed test manifests, 256 seeds each: {args.output}')


if __name__ == '__main__':
    main()
