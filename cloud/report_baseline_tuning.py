"""Summarize a baseline tuning run (tune_astar.py, or tune_apf.py on feature/apf):
markdown tables + figure. Accepts one run directory, or a root with one
sub-directory per scenario (Slurm array layout)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path(sys.argv[1])
SCEN = ['10_sparse', '10_dense', '20_sparse', '20_dense', '20_sparse_dynamic']
LABEL = {'10_sparse': 'E1 10_sparse', '10_dense': 'E2 10_dense', '20_sparse': 'E3 20_sparse',
         '20_dense': 'E4 20_dense', '20_sparse_dynamic': 'E_dyn'}
def locate(scenario, name):
    # Single-directory runs (local) or one sub-directory per scenario (Slurm array).
    for path in (run / name, run / scenario / name):
        if path.exists():
            return path
    return None


res = {s: json.loads(locate(s, f'{s}_result.json').read_text()) for s in SCEN if locate(s, f'{s}_result.json')}
plan = json.loads(next(p for p in [run / 'plan.json', *[run / s / 'plan.json' for s in SCEN]] if p.exists()).read_text())
params = {c['id']: c['params'] for c in plan['candidates']}


def pct(x):
    return f'{100 * x:.1f}%'


def ci(s):
    lo, hi = s['success_ci95']
    return f'{100 * lo:.1f}–{100 * hi:.1f}'


def t(x):
    return '–' if x is None else f'{x:.2f}'


lines = ['## Validation results (128 maps per scenario, same machine)', '',
         '| Scenario | Config | Success (95% CI) | Collision | Crash | Timeout | Time to goal (s) | Path (m) | SPL |',
         '|---|---|---|---|---|---|---|---|---|']
for s, r in res.items():
    for name, key in [('incumbent', 'incumbent_summary'), (f"tuned ({r['selected']})", 'selected_summary')]:
        m = r[key]
        lines.append(f"| {LABEL[s]} | {name} | {m['success_count']}/128 {pct(m['success_rate'])} ({ci(m)}) | "
                     f"{pct(m['collision_rate'])} | {pct(m['crash_rate'])} | {pct(m['timeout_rate'])} | "
                     f"{t(m['success_time_mean'])} | {t(m['success_path_mean'])} | {m['spl']:.3f} |")
lines += ['', '## Selected parameters', '',
          '| Scenario | id | ' + ' | '.join(plan['search_space']) + ' |',
          '|---|---|' + '---|' * len(plan['search_space'])]
keys = list(plan['search_space'])
lines.append('| (incumbent) | c00_incumbent | ' + ' | '.join(str(params['c00_incumbent'][k]) for k in keys) + ' |')
for s, r in res.items():
    lines.append(f"| {LABEL[s]} | {r['selected']} | " + ' | '.join(str(r['selected_params'][k]) for k in keys) + ' |')
lines += ['', '## Successive-halving trace (successes / maps seen)', '']
for s, r in res.items():
    lines.append(f'**{LABEL[s]}**')
    for h in r['rungs']:
        top = ', '.join(f"{x['id']} {x['success_count']}" for x in h['ranking'][:8])
        lines.append(f"- rung {h['rung']} ({h['maps']} maps): {top}")
    lines.append('')
(run / 'results_tables.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

fig, ax = plt.subplots(figsize=(8, 3.6))
xs = range(len(res))
for off, key, name, color in [(-0.18, 'incumbent_summary', 'Incumbent', '#9aa5b1'),
                              (0.18, 'selected_summary', 'Per-scenario tuned', '#2b6cb0')]:
    vals = [r[key]['success_rate'] * 100 for r in res.values()]
    los = [v - r[key]['success_ci95'][0] * 100 for v, r in zip(vals, res.values())]
    his = [r[key]['success_ci95'][1] * 100 - v for v, r in zip(vals, res.values())]
    ax.bar([x + off for x in xs], vals, 0.34, yerr=[los, his], capsize=3, label=name, color=color)
ax.set_xticks(list(xs), [LABEL[s] for s in res])
ax.set_ylabel('Validation success (%)')
ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(run / 'validation_success.png', dpi=160)
