"""Plot one recorded episode without loading a model or simulator."""
import argparse
from pathlib import Path
from analyze_diagnostics import resolve_directory, load_episode


def plot(directory, seed, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    directory = resolve_directory(directory)
    matches = list(directory.glob(f'*_seed_{seed}.episode.json'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one episode for seed {seed}, found {len(matches)}')
    episode, geometry, rows = load_episode(matches[0])
    state = np.asarray([r['state'] for r in rows]); time = np.asarray([r['time_s'] for r in rows])
    command = np.asarray([r['target_velocity'] if r['target_velocity'] is not None else [np.nan]*3 for r in rows])
    if len(command) > 1:
        command[0] = command[1]  # Display the first interval's command back to t=0.
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), layout='constrained')
    ax = axes[0, 0]
    for x, y, wx, wy, height in geometry['static_boxes']:
        ax.add_patch(Rectangle((x-wx/2, y-wy/2), wx, wy, color='gray', alpha=.35))
    for x, y, radius, *_ in geometry['dynamic_cylinders']:
        ax.add_patch(Circle((x, y), radius, fill=False, color='purple', linestyle='--'))
    if geometry['dynamic_cylinders']:
        dynamic = np.asarray([r['dynamic_cylinders'] for r in rows])
        for i in range(dynamic.shape[1]):
            ax.plot(dynamic[:, i, 0], dynamic[:, i, 1], color='purple', alpha=.3)
        ax.plot([], [], '--', color='purple', label='Moving obstacles: initial outline / path')
    ax.plot(state[:, 0], state[:, 1], color='tab:blue', label='Flight path')
    ax.scatter(*geometry['start'][:2], marker='o', color='green', label='Start')
    ax.scatter(*geometry['goal'][:2], marker='*', color='green', s=100, label='Goal')
    ax.scatter(*state[-1, :2], marker='x', color='red', s=70, label='Terminal')
    ax.set(xlim=geometry['bounds'][0], ylim=geometry['bounds'][1], xlabel='x (m)', ylabel='y (m)', title='Top view (obstacle footprints)')
    ax.set_aspect('equal'); ax.legend(fontsize=8)
    ax = axes[0, 1]; ax.plot(time, state[:, 2], label='Altitude')
    for z in geometry['bounds'][2]: ax.axhline(z, color='red', linestyle='--')
    ax.set(title='Altitude and permitted bounds', ylabel='z (m)')
    ax = axes[1, 0]
    ax.plot(time, np.linalg.norm(state[:, 10:13], axis=1), label='Actual speed')
    # Command at row t applied during (t-1, t], hence steps-pre.
    ax.step(time, np.linalg.norm(command, axis=1), where='pre', label='Command speed', alpha=.8)
    ax.set(title='Speed magnitude', ylabel='m/s'); ax.legend()
    ax = axes[1, 1]
    for i, name in enumerate(['x', 'y', 'z']):
        ax.plot(time, state[:, 10+i], label=f'Actual {name}')
        ax.step(time, command[:, i], where='pre', linestyle='--', alpha=.6, label=f'Command {name}')
    ax.set(title='Velocity components', ylabel='m/s'); ax.legend(fontsize=8, ncol=2)
    ax = axes[2, 0]
    clearance = np.asarray([min((x['clearance'] for x in r['nearby_obstacles']), default=np.nan) for r in rows])
    radii = np.asarray([r['clearance_query_radius'] for r in rows])
    ax.plot(time, clearance, label='Measured surface clearance')
    unknown = np.isnan(clearance)
    ax.scatter(time[unknown], radii[unknown], marker='^', s=10, color='gray', label='Outside query radius (lower bound)')
    ax.axhline(geometry['collision_distance'], color='red', linestyle='--', label='Collision threshold')
    ax.set(title='Obstacle clearance (gaps are unmeasured)', ylabel='m'); ax.legend(fontsize=8)
    ax = axes[2, 1]
    ax.plot(time, state[:, 7], label='Roll'); ax.plot(time, state[:, 8], label='Pitch')
    ax.axhline(1.2, color='red', linestyle='--'); ax.axhline(-1.2, color='red', linestyle='--')
    ax.set(title='Attitude and termination limits', ylabel='rad'); ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax is not axes[0, 0]: ax.set_xlabel('Time (s)')
    flags = [k for k in ['success', 'collision', 'crash', 'timeout'] if episode['outcome'][k]]
    fig.suptitle(f"Seed {seed} | {' + '.join(flags)} | {time[-1]:.2f} s")
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150); plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input / 'plots' / f'seed_{args.seed}.png'
    print(plot(args.input, args.seed, output))


if __name__ == '__main__': main()
