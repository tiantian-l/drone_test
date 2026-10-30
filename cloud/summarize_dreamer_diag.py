"""Crash split for all thesis DreamerV3 test pairs (RQ1-RQ3 tables).

  python cloud/summarize_dreamer_diag.py [--output dreamer_crash_split.md]

Run after `ALL=1 bash cloud/submit_dreamer_diag.sh` has finished. For every
(policy, test scenario) pair it reads
  final_eval/<scenario>/job_*/*__<ckpt>__episodes.jsonl        thesis test run
  final_eval_diag/<scenario>/<label>*/*__<ckpt>__episodes.jsonl diagnostics re-run
and reports the thesis-run rates, with crashes split into
  out of bounds = terminal position outside the flight region (any axis)
  attitude      = remaining crashes (roll or pitch beyond 1.2 rad)
The split is taken from the re-run only for maps whose outcome is identical in
both runs; any crash that cannot be classified is listed as 'unknown'.
"""
import argparse
import json
from pathlib import Path

from compare_baselines import LOGDIR, NAME, flag, rows_of

CKPT = {
    'pi1': 'step_0002300000_success_0.9766',
    'pi2': 'step_0001700000_success_0.9375',
    'pi3': 'step_0002200000_success_0.9531',
    'pi4': 'step_0004800000_success_0.9297',
    'pi_dyn': 'step_0002900000_success_0.8984',
    'pi_scratch': 'step_0005200000_success_0.9297',
}
STATIC = ['10_sparse', '10_dense', '20_sparse', '20_dense']
PAIRS = ([(p, s) for p in ('pi1', 'pi2', 'pi3', 'pi4') for s in STATIC]            # Tables 4.2/4.4
         + [(p, s) for p in ('pi3', 'pi_dyn', 'pi_scratch')
            for s in ('20_sparse', '20_sparse_dynamic')])                          # Table 4.5
OUTCOMES = ('success', 'collision', 'crash', 'timeout')


def newest(paths):
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def pair(root, diag_root, label, scen):
    stem = CKPT[label]
    ep = newest((root / scen).glob(f'job_*/*__{stem}__episodes.jsonl'))
    if ep is None:
        return dict(error='thesis run missing')
    rows = rows_of(ep)
    summary = json.loads(Path(str(ep).replace('__episodes.jsonl', '__summary.json')).read_text())
    res = dict(rows=rows, summary=summary, source=ep, split=None, agree=None, diag=None)
    d = newest((diag_root / scen).glob(f'{label}*/*__{stem}__episodes.jsonl'))
    if d is None:
        return res
    drows = rows_of(d)
    events = {}
    for f in Path(str(d).replace('__episodes.jsonl', '__diagnostics')).glob('*.episode.json'):
        e = json.loads(f.read_text())
        events[int(e['seed'])] = e['terminal']['events']
    same = {s for s in rows if s in drows and all(flag(rows[s], k) == flag(drows[s], k) for k in OUTCOMES)}
    oob = att = unk = 0
    for s, r in rows.items():
        if not flag(r, 'crash'):
            continue
        ev = events.get(s) if s in same else None
        if ev is None:
            unk += 1
        elif ev['horizontal_oob'] or ev['below_floor'] or ev['above_ceiling']:
            oob += 1
        else:
            att += 1
    res.update(split=(oob, att, unk), agree=len(same) / len(rows), diag=d)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dreamer-root', type=Path, default=LOGDIR / 'final_eval')
    ap.add_argument('--dreamer-diag-root', type=Path, default=LOGDIR / 'final_eval_diag')
    ap.add_argument('--output', type=Path, default=Path('dreamer_crash_split.md'))
    args = ap.parse_args()
    out = ['# DreamerV3 test results with crash split (256 maps, thesis environment)', '',
           '| Policy | Test | Success | Collision | Static coll. | Moving coll. | Crash | Out of bounds '
           '| Attitude | Unknown | Timeout | Time (s) | Re-run agreement |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    sources = []
    for label, scen in PAIRS:
        r = pair(args.dreamer_root, args.dreamer_diag_root, label, scen)
        if 'error' in r:
            out.append(f'| {label} | {NAME[scen]} | {r["error"]} |' + ' |' * 11)
            continue
        rows = list(r['rows'].values())
        n = len(rows)
        pc = lambda k: f'{100 * sum(flag(x, k) for x in rows) / n:.1f}'
        if r['split'] is None:
            o = a = u = 'pending'
            agree = 'no re-run'
        else:
            o, a, u = (f'{100 * v / n:.1f}' for v in r['split'])
            agree = f'{100 * r["agree"]:.1f}%'
            sources.append(f'- {label} {NAME[scen]}: `{r["diag"]}`')
        dyn = scen.endswith('dynamic')
        sc = pc('static_collision') if dyn else '–'
        mc = pc('dynamic_collision') if dyn else '–'
        t = r['summary'].get('successful_time_seconds_mean')
        out.append(f"| {label} | {NAME[scen]} | {pc('success')} | {pc('collision')} | {sc} | {mc} | {pc('crash')} "
                   f"| {o} | {a} | {u} | {pc('timeout')} | {'–' if t is None else f'{t:.2f}'} | {agree} |")
    out += ['', 'Rates in % of 256 episodes. Rates come from the thesis run; the split comes from the',
            'diagnostics re-run for maps with identical outcomes. Out of bounds + Attitude + Unknown = Crash.',
            '', 'Diagnostics sources:'] + sources
    text = '\n'.join(out) + '\n'
    args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
