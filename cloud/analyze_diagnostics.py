"""Validate full evaluation captures and export episode/failure tables (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

EVENTS = ('horizontal_oob', 'below_floor', 'above_ceiling', 'roll_limit', 'pitch_limit')
OUTCOMES = ('success', 'collision', 'crash', 'timeout', 'static_collision', 'dynamic_collision')


def read(path):
    return json.loads(Path(path).read_text())


def resolve_directory(path):
    path = Path(path)
    if (path / 'run.json').is_file():
        return path
    matches = list(path.glob('*__diagnostics'))
    if len(matches) != 1:
        raise ValueError('Pass one diagnostics directory or a result directory containing exactly one.')
    return matches[0]


def load_episode(path):
    episode = read(path)
    geometry_path = Path(str(path).replace('.episode.json', '.map.json'))
    geometry = read(geometry_path)
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    if digest != episode['geometry_sha256'] or geometry['seed'] != episode['seed']:
        raise ValueError(f'Geometry identity mismatch: {path}')
    trajectory = episode.get('trajectory')
    if not trajectory or Path(trajectory).name != trajectory:
        raise ValueError(f'Full trajectory required: {path}')
    rows = [json.loads(line) for line in (path.parent / trajectory).read_text().splitlines() if line.strip()]
    if len(rows) != episode['steps'] + 1 or not rows[0]['is_first'] or not rows[-1]['is_last']:
        raise ValueError(f'Incomplete trajectory: {path}')
    for i, row in enumerate(rows):
        if row['step'] != i or not math.isclose(row['time_s'], i / geometry['ctrl_freq'], abs_tol=1e-7):
            raise ValueError(f'Time/step mismatch: {path}, row {i}')
        if row['is_first'] != (i == 0) or row['is_last'] != (i == len(rows)-1):
            raise ValueError(f'Episode boundary mismatch: {path}')
        if len(row['state']) < 16 or not all(math.isfinite(x) for x in row['state']):
            raise ValueError(f'Invalid state: {path}')
        if i and (row['action_executed'] is None or row['target_velocity'] is None):
            raise ValueError(f'Missing executed action: {path}')
    if rows[-1]['state'] != episode['terminal']['state'] or rows[-1]['events'] != episode['terminal']['events']:
        raise ValueError(f'Terminal mismatch: {path}')
    length = sum(math.dist(a['state'][:3], b['state'][:3]) for a, b in zip(rows, rows[1:]))
    if not math.isclose(length, episode['path_length'], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f'Path length mismatch: {path}')
    return episode, geometry, rows


def feature_row(path, episode, geometry, rows):
    terminal = episode['terminal']
    outcome = episode['outcome']
    nearby = terminal['nearby_obstacles']
    collisions = [x for x in nearby if x['clearance'] < geometry['collision_distance']]
    measured = [x['clearance'] for r in rows for x in r['nearby_obstacles']]
    speeds = [math.sqrt(sum(v*v for v in r['state'][10:13])) for r in rows[1:]]
    labels = [k for k in ('collision', 'crash', 'timeout') if outcome[k]]
    flags = {k: bool(terminal['events'][k]) for k in EVENTS}
    labels += [k for k, v in flags.items() if v]
    if not outcome['success'] and not labels:
        labels = ['unclassified_failure']
    return dict(seed=episode['seed'], worker=episode['worker'], episode=episode['episode'],
        **{k: int(bool(outcome[k])) for k in OUTCOMES}, **flags,
        terminal_labels=';'.join(labels), behavior_label='unreviewed',
        collision_body_ids=';'.join(str(x['body_id']) for x in collisions),
        overlap_body_ids=';'.join(str(x['body_id']) for x in nearby if x['clearance'] < 0),
        steps=episode['steps'], duration_s=rows[-1]['time_s'], path_length_m=episode['path_length'],
        start_goal_distance_m=math.dist(geometry['start'], geometry['goal']),
        static_obstacles=len(geometry['static_boxes']), dynamic_obstacles=len(geometry['dynamic_cylinders']),
        final_goal_distance_m=math.dist(rows[-1]['state'][:3], geometry['goal']),
        mean_speed_mps=sum(speeds)/len(speeds) if speeds else None,
        min_measured_clearance_m=min(measured) if measured else None,
        clearance_unobserved_steps=sum(not r['nearby_obstacles'] for r in rows),
        max_abs_roll_rad=max(abs(r['state'][7]) for r in rows),
        max_abs_pitch_rad=max(abs(r['state'][8]) for r in rows),
        episode_file=path.name, trajectory_file=episode['trajectory'])


def analyze(directory):
    directory = resolve_directory(directory)
    run = read(directory / 'run.json')
    if run.get('status') != 'complete' or run.get('mode') != 'full':
        raise ValueError('A complete full-mode capture is required.')
    paths = sorted(directory.glob('*.episode.json'))
    expected = run['episodes']
    if len(paths) != expected or len(list(directory.glob('*.map.json'))) != expected or len(list(directory.glob('*.trajectory.jsonl'))) != expected:
        raise ValueError('Episode/map/trajectory file count mismatch.')
    records, seeds, identities = [], set(), set()
    for path in paths:
        episode, geometry, rows = load_episode(path)
        identity = (episode['worker'], episode['episode'])
        if episode['seed'] in seeds or identity in identities:
            raise ValueError('Duplicate seed or worker/episode.')
        seeds.add(episode['seed']); identities.add(identity)
        records.append(feature_row(path, episode, geometry, rows))
    expected_ids = {(w, i) for w in range(run['workers']) for i in range(run['episodes_per_worker'])}
    if identities != expected_ids or expected != len(expected_ids):
        raise ValueError('Worker quota mismatch.')
    counts = {k: sum(r[k] for r in records) for k in OUTCOMES + EVENTS}
    # Test evaluator uses a matching filename prefix; training uses cycle audit files.
    if directory.name.endswith('__diagnostics'):
        summary_path = directory.parent / (directory.name[:-len('diagnostics')] + 'summary.json')
    elif run.get('source') == 'training_validation':
        summary_path = directory.parent.parent / f"evaluation_{run['evaluation_protocol']}_{expected}maps" / f"cycle_{run['eval_cycle']}_summary.json"
    else:
        summary_path = directory.parent / 'summary.json'
    compared = False
    if summary_path.exists():
        reference = read(summary_path)
        if reference['episodes'] != expected:
            raise ValueError('Reference episode count differs.')
        for key in OUTCOMES:
            if key + '_count' in reference and reference[key + '_count'] != counts[key]:
                raise ValueError(f'Reference count differs: {key}')
        # Verify individual map outcomes too when the original episode file exists.
        episode_reference = summary_path.with_name(summary_path.name.replace('summary.json', 'episodes.jsonl'))
        if episode_reference.exists():
            reference_rows = [json.loads(x) for x in episode_reference.read_text().splitlines() if x.strip()]
            reference_by_seed = {int(r['eval_map_seed']): r for r in reference_rows}
            if len(reference_rows) != expected or set(reference_by_seed) != seeds:
                raise ValueError('Reference per-map coverage differs.')
            for row in records:
                ref = reference_by_seed[row['seed']]
                for key in OUTCOMES:
                    if key in ref and bool(ref[key]) != bool(row[key]):
                        raise ValueError(f'Per-map outcome differs: seed {row["seed"]}, {key}')
        compared = True
    failures = [r for r in records if not r['success']]
    summary = dict(episodes=expected, failures=len(failures), counts=counts,
        rates={k: v/expected for k, v in counts.items()},
        reference_summary_checked=compared, reference_summary=str(summary_path),
        source=str(directory.resolve()), metadata=run,
        notes=['Terminal flags may overlap; behavior labels require review.',
               'Clearance outside the finite query radius is unknown, not infinity.',
               'Geometric overlap is not a contact-force measurement.'])
    return records, failures, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory; default INPUT/analysis')
    args = parser.parse_args()
    records, failures, summary = analyze(args.input)
    output = args.output or args.input / 'analysis'
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('episodes.csv', records), ('failures.csv', failures)]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(values)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"Validated {len(records)} episodes; {len(failures)} failures. Results: {output}")


if __name__ == '__main__':
    main()
