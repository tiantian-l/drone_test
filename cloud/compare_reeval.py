"""Compare the PID-reset re-evaluation of the DreamerV3 checkpoints with the
original (pre-fix) test results, and print the updated thesis tables.

  python cloud/compare_reeval.py \
      [--new-root /projects/.../logdir/final_eval_pidreset_v2] \
      [--old-root /projects/.../logdir/final_eval] [--output reeval_report.md]

Per (policy, test scenario): old vs new success on the same 256 maps, maps
gained/lost, exact McNemar p-value, failures within the first second
(episode_length <= 31 control steps, the window in which stale PID state acts),
and whether the initial map geometry is identical (geometry_sha256).
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = Path('/projects/EEHPC-DEV-2026D07-102/dreamer/logdir')
STATIC = ['10_sparse', '10_dense', '20_sparse', '20_dense']
NAME = {'10_sparse': 'E1', '10_dense': 'E2', '20_sparse': 'E3', '20_dense': 'E4', '20_sparse_dynamic': 'E_dyn'}
MATRIX = ([(p, s) for p in ('pi1', 'pi2', 'pi3', 'pi4') for s in STATIC] + [('pi3', '20_sparse_dynamic')] +
          [(p, s) for p in ('pi_dyn', 'pi_scratch') for s in ('20_sparse', '20_sparse_dynamic')])
EARLY_STEPS = 31


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(prefix_glob, base):
    eps = sorted(base.glob(prefix_glob + '__episodes.jsonl'), key=lambda p: p.stat().st_mtime)
    if not eps:
        return None
    ep = eps[-1]
    summary = Path(str(ep).replace('__episodes.jsonl', '__summary.json'))
    rows = {int(r['eval_map_seed']): r for r in map(json.loads, ep.read_text().splitlines())}
    return dict(rows=rows, summary=json.loads(summary.read_text()), path=ep, copies=len(eps))


def ok(r):
    # Records store outcome flags as bool or as 0.0/1.0 floats.
    return bool(float(r['success']))


def failed_early(r):
    return (not ok(r)) and int(float(r['episode_length'])) <= EARLY_STEPS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', type=Path, default=ROOT / 'cloud/dreamer_checkpoints.tsv')
    ap.add_argument('--new-root', type=Path, default=LOGDIR / 'final_eval_pidreset_v2')
    ap.add_argument('--old-root', type=Path, default=LOGDIR / 'final_eval')
    ap.add_argument('--output', type=Path, default=Path('reeval_report.md'))
    args = ap.parse_args()
    stems = {}
    for line in args.table.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            label, _, _, ckpt = line.split('\t')[:4]
            stems[label] = Path(ckpt).name.removesuffix('.ckpt')
    new, old = {}, {}
    for label, scen in MATRIX:
        new[label, scen] = load(f'{label}*/*', args.new_root / scen)  # includes <label>_retryN
        old[label, scen] = load(f'job_*/*__{stems[label]}', args.old_root / scen)
    out = ['# DreamerV3 re-evaluation after the PID-reset fix', '',
           f'New: `{args.new_root}`  Old: `{args.old_root}`', '',
           '## Old vs new (same 256 test maps)', '',
           '| Policy | Test | Old success | New success | Δ (pp) | Gained | Lost | McNemar p | '
           f'Early fails old/new (≤{EARLY_STEPS} steps) | Same geometry |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for key in MATRIX:
        n, o = new[key], old[key]
        if n is None:
            out.append(f'| {key[0]} | {NAME[key[1]]} | – | missing | | | | | | |')
            continue
        ns = sum(ok(r) for r in n['rows'].values()) / len(n['rows'])
        if o is None:
            out.append(f'| {key[0]} | {NAME[key[1]]} | not found | {100*ns:.1f} | | | | | – / '
                       f"{sum(map(failed_early, n['rows'].values()))} | |")
            continue
        seeds = sorted(set(n['rows']) & set(o['rows']))
        os_ = sum(ok(o['rows'][s]) for s in seeds) / len(seeds)
        gained = sum(ok(n['rows'][s]) and not ok(o['rows'][s]) for s in seeds)
        lost = sum(ok(o['rows'][s]) and not ok(n['rows'][s]) for s in seeds)
        geo = n['summary'].get('geometry_sha256') == o['summary'].get('geometry_sha256')
        early = (sum(failed_early(o['rows'][s]) for s in seeds), sum(failed_early(n['rows'][s]) for s in seeds))
        note = '' if len(seeds) == 256 else f' ({len(seeds)} common maps)'
        out.append(f'| {key[0]} | {NAME[key[1]]} | {100*os_:.1f} | {100*ns:.1f}{note} | {100*(ns-os_):+.1f} | '
                   f'{gained} | {lost} | {mcnemar(gained, lost):.3f} | {early[0]} / {early[1]} | '
                   f"{'yes' if geo else 'NO'} |")
    out += ['', 'Several old result copies for one pair: the newest file is used.', '']

    def cell(label, scen, key, scale=100, fmt='{:.1f}'):
        r = new[label, scen]
        if r is None or r['summary'].get(key) is None:
            return '–'
        return fmt.format(r['summary'][key] * scale)

    out += ['## Updated Table 4.3 (success %, new protocol)', '',
            '| Policy | ' + ' | '.join(NAME[s] for s in STATIC) + ' |', '|---|---|---|---|---|']
    for p in ('pi1', 'pi2', 'pi3', 'pi4'):
        out.append(f'| {p} | ' + ' | '.join(cell(p, s, 'success_rate') for s in STATIC) + ' |')
    out += ['', '## Updated Tables 4.2 / 4.4 (success, collision, crash, timeout %, time to goal s)', '',
            '| Policy | Test | Success | Collision | Crash | Timeout | Time |', '|---|---|---|---|---|---|---|']
    for p in ('pi1', 'pi2', 'pi3', 'pi4'):
        for s in STATIC:
            out.append(f'| {p} | {NAME[s]} | ' + ' | '.join(cell(p, s, k) for k in
                       ('success_rate', 'collision_rate', 'crash_rate', 'timeout_rate')) +
                       f" | {cell(p, s, 'successful_time_seconds_mean', 1, '{:.2f}')} |")
    out += ['', '## Updated Tables 4.5 / 4.6 (E3 and E_dyn)', '',
            '| Test | Policy | Success | Collision | Crash | Timeout | Time | Static coll. | Moving coll. |',
            '|---|---|---|---|---|---|---|---|---|']
    for s in ('20_sparse', '20_sparse_dynamic'):
        for p in ('pi3', 'pi_dyn', 'pi_scratch'):
            out.append(f'| {NAME[s]} | {p} | ' + ' | '.join(cell(p, s, k) for k in
                       ('success_rate', 'collision_rate', 'crash_rate', 'timeout_rate')) +
                       f" | {cell(p, s, 'successful_time_seconds_mean', 1, '{:.2f}')} | "
                       f"{cell(p, s, 'static_collision_rate')} | {cell(p, s, 'dynamic_collision_rate')} |")
    text = '\n'.join(out) + '\n'
    args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
