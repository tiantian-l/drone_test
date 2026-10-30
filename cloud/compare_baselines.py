"""DreamerV3 vs Map-A* on the same 256 test maps per condition (thesis environment).

  python cloud/compare_baselines.py [--output baseline_comparison.md]

Inputs (defaults under /projects/EEHPC-DEV-2026D07-102/dreamer/logdir):
  final_eval/<scenario>/job_*/            DreamerV3 thesis test results (newest copy per checkpoint)
  final_eval_diag/<scenario>/<label>*/    optional re-run with EVAL_DIAGNOSTICS=summary
                                          (cloud/submit_dreamer_diag.sh) for the crash split
  baseline_test/astar/<scenario>_job_*/           Map-A* with flight region
  baseline_test/astar_noregion/<scenario>_job_*/  Map-A* observation only
Only Map-A* runs in the thesis environment are used (nav_aviary.py without the
per-episode PID reset, identified by its source hash in config.json); if several
remain, the newest is used.

Crash split:
  * out of bounds  = terminal position outside the flight region (any axis)
  * attitude       = remaining crashes (roll or pitch beyond 1.2 rad)
  DreamerV3: from the diagnostics terminal events (exact). Map-A*: from the last
  logged position extrapolated by one control step against the flight region.
Paired comparison: exact McNemar test on per-map success, DreamerV3 vs each A*.
"""
import argparse
import json
import math
from pathlib import Path

LOGDIR = Path('/projects/EEHPC-DEV-2026D07-102/dreamer/logdir')
THESIS_ENV_SHA = 'c953b9e81fed'  # drone_nav/envs/nav_aviary.py without PID reset
SCEN = ['10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic']
NAME = dict(zip(SCEN, ['E1', 'E2', 'E3', 'E4', 'E_dyn']))
DREAMER = {  # scenario -> (label, checkpoint stem) used in the comparison
    '10_sparse': ('pi1', 'step_0002300000_success_0.9766'),
    '10_dense': ('pi2', 'step_0001700000_success_0.9375'),
    '20_sparse': ('pi3', 'step_0002200000_success_0.9531'),
    '20_dense': ('pi4', 'step_0004800000_success_0.9297'),
    '20_sparse_dynamic': ('pi_scratch', 'step_0005200000_success_0.9297'),
}


def flag(r, key):
    return bool(float(r.get(key, 0) or 0))


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def rows_of(path):
    return {int(r['eval_map_seed']): r for r in map(json.loads, Path(path).read_text().splitlines())}


def dreamer(root, diag_root, scen):
    label, stem = DREAMER[scen]
    eps = sorted((root / scen).glob(f'job_*/*__{stem}__episodes.jsonl'), key=lambda p: p.stat().st_mtime)
    if not eps:
        return None
    ep = eps[-1]
    rows = rows_of(ep)
    summary = json.loads(Path(str(ep).replace('__episodes.jsonl', '__summary.json')).read_text())
    split, agree = None, None
    diag_eps = sorted((diag_root / scen).glob(f'{label}*/*__{stem}__episodes.jsonl'), key=lambda p: p.stat().st_mtime)
    if diag_eps:
        d = diag_eps[-1]
        drows = rows_of(d)
        events = {}
        for f in Path(str(d).replace('__episodes.jsonl', '__diagnostics')).glob('*.episode.json'):
            e = json.loads(f.read_text())
            events[int(e['seed'])] = e['terminal']['events']
        same = [s for s in rows if s in drows and
                all(flag(rows[s], k) == flag(drows[s], k) for k in ('success', 'collision', 'crash', 'timeout'))]
        agree = len(same) / len(rows)
        oob = att = 0
        for s, r in rows.items():  # thesis run outcomes; split from the re-run where outcomes agree
            if not flag(r, 'crash'):
                continue
            ev = events.get(s) if s in same else None
            if ev is None:
                continue
            if ev['horizontal_oob'] or ev['below_floor'] or ev['above_ceiling']:
                oob += 1
            else:
                att += 1
        unknown = sum(flag(r, 'crash') for r in rows.values()) - oob - att
        split = (oob, att, unknown)
    return dict(rows=rows, summary=summary, split=split, agree=agree, source=ep)


def astar(root, scen):
    runs = []
    for d in (root).glob(f'{scen}_job_*'):
        cfg = d / 'config.json'
        if not (cfg.is_file() and (d / 'episodes.jsonl').is_file()):
            continue
        c = json.loads(cfg.read_text())
        if c['sources']['drone_nav/envs/nav_aviary.py'].startswith(THESIS_ENV_SHA):
            runs.append((d.stat().st_mtime, d, c))
    if not runs:
        return None
    _, d, c = max(runs)
    rows = rows_of(d / 'episodes.jsonl')
    summary = json.loads((d / 'summary.json').read_text())
    b = c['environment']['bounds']
    lo, hi = [b[0], b[2], b[4]], [b[1], b[3], b[5]]
    dt = 1.0 / float(c['environment'].get('ctrl_freq', 30))
    last = {}
    with (d / 'trajectories.jsonl').open() as fh:
        for line in fh:
            t = json.loads(line)
            last[int(t['seed'])] = t
    oob = att = 0
    for s, r in rows.items():
        if not flag(r, 'crash'):
            continue
        t = last[s]
        p = [x + v * dt for x, v in zip(t['position'], t['velocity'])]
        if any(p[i] < lo[i] + 0.02 or p[i] > hi[i] - 0.02 for i in range(3)):
            oob += 1
        else:
            att += 1
    return dict(rows=rows, summary=summary, split=(oob, att, 0), source=d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dreamer-root', type=Path, default=LOGDIR / 'final_eval')
    ap.add_argument('--dreamer-diag-root', type=Path, default=LOGDIR / 'final_eval_diag')
    ap.add_argument('--astar-root', type=Path, default=LOGDIR / 'baseline_test/astar')
    ap.add_argument('--astar-none-root', type=Path, default=LOGDIR / 'baseline_test/astar_noregion')
    ap.add_argument('--output', type=Path, default=Path('baseline_comparison.md'))
    args = ap.parse_args()
    methods = [('DreamerV3', lambda s: dreamer(args.dreamer_root, args.dreamer_diag_root, s)),
               ('Map-A* (+ region)', lambda s: astar(args.astar_root, s)),
               ('Map-A* (obs. only)', lambda s: astar(args.astar_none_root, s))]
    data = {(m, s): f(s) for m, f in methods for s in SCEN}
    out = ['# DreamerV3 vs Map-A* (256 test maps per condition, thesis environment)', '',
           '| Test | Method | Success ↑ | Collision ↓ | Out of bounds ↓ | Attitude ↓ | Timeout ↓ | Time to goal (s) ↓ |',
           '|---|---|---|---|---|---|---|---|']
    notes = []
    for s in SCEN:
        for m, _ in methods:
            d = data[m, s]
            if d is None:
                out.append(f'| {NAME[s]} | {m} | missing | | | | | |')
                continue
            rows = list(d['rows'].values())
            n = len(rows)
            pc = lambda k: 100 * sum(flag(r, k) for r in rows) / n
            if d['split'] is None:
                o = a = '?'
                notes.append(f'{NAME[s]} {m}: crash split unavailable (no diagnostics re-run)')
            else:
                oob, att, unk = d['split']
                o, a = f'{100*oob/n:.1f}', f'{100*att/n:.1f}'
                if unk:
                    notes.append(f'{NAME[s]} {m}: {unk} crash(es) not classified (outcome differed in re-run)')
            t = d['summary'].get('successful_time_seconds_mean')
            out.append(f"| {NAME[s]} | {m} | {pc('success'):.1f} | {pc('collision'):.1f} | {o} | {a} | "
                       f"{pc('timeout'):.1f} | {'–' if t is None else f'{t:.2f}'} |")
            if m == 'DreamerV3' and d['agree'] is not None:
                notes.append(f"{NAME[s]} DreamerV3: diagnostics re-run reproduces {100*d['agree']:.1f}% of per-map outcomes")
    out += ['', '## Paired comparison of success (exact McNemar)', '',
            '| Test | Comparison | DreamerV3 only | A* only | Both | Neither | p |', '|---|---|---|---|---|---|---|']
    for s in SCEN:
        dr = data['DreamerV3', s]
        for m, _ in methods[1:]:
            a = data[m, s]
            if dr is None or a is None:
                continue
            seeds = sorted(set(dr['rows']) & set(a['rows']))
            x = [(flag(dr['rows'][k], 'success'), flag(a['rows'][k], 'success')) for k in seeds]
            b, c = sum(p and not q for p, q in x), sum(q and not p for p, q in x)
            both, neither = sum(p and q for p, q in x), sum(not p and not q for p, q in x)
            note = '' if len(seeds) == 256 else f' ({len(seeds)} maps)'
            out.append(f'| {NAME[s]} | DreamerV3 vs {m}{note} | {b} | {c} | {both} | {neither} | {mcnemar(b, c):.3g} |')
    out += ['', 'Notes:'] + [f'- {x}' for x in notes] + ['', 'Sources:'] + \
           [f"- {m} {NAME[s]}: `{data[m, s]['source']}`" for m, _ in methods for s in SCEN if data[m, s]]
    text = '\n'.join(out) + '\n'
    args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
