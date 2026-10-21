"""Sequentially plot and replay every unsuccessful recorded episode."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from analyze_diagnostics import read, resolve_directory


def failure_seeds(directory):
    run = read(directory / 'run.json')
    if run.get('status') != 'complete' or run.get('mode') != 'full':
        raise ValueError('A complete full-mode diagnostics capture is required.')
    paths = sorted(directory.glob('*.episode.json'))
    if len(paths) != run['episodes']:
        raise ValueError('Diagnostic episode count does not match run.json')
    seeds, failures = set(), []
    for path in paths:
        row = read(path)
        seed = int(row['seed'])
        if seed in seeds:
            raise ValueError(f'Duplicate seed: {seed}')
        seeds.add(seed)
        if not bool(row['outcome']['success']):
            failures.append(seed)
    return sorted(failures)


def complete(path, kind):
    if not path.is_file() or not path.stat().st_size:
        return False
    if kind == 'video':
        metadata = path.with_suffix('.camera.json')
        if not metadata.is_file():
            return False
        try:
            return read(metadata).get('frames', 0) > 0
        except (ValueError, OSError):
            return False
    return True


def run_batch(source, output, fps=20, speed=1., width=1280, skip_existing=True):
    directory = resolve_directory(source).resolve()
    seeds = failure_seeds(directory)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    report = output / 'batch.jsonl'
    errors = 0
    print(f'{len(seeds)} failure cases; output: {output}', flush=True)
    for index, seed in enumerate(seeds, 1):
        for kind, script, suffix, subdir in [
                ('plot', 'plot_diagnostic_episode.py', '.png', 'plots'),
                ('video', 'replay_diagnostic_episode.py', '.mp4', 'videos')]:
            target = output / subdir / f'seed_{seed}{suffix}'
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(scripts / script), str(directory),
                       '--seed', str(seed), '--output', str(target)]
            if kind == 'video':
                command += ['--fps', str(fps), '--speed', str(speed), '--width', str(width)]
            entry = dict(time=datetime.now(timezone.utc).isoformat(), seed=seed,
                         kind=kind, source=str(directory), output=str(target), command=command)
            print(f'[{index}/{len(seeds)}] seed {seed}: {kind}', flush=True)
            if skip_existing and complete(target, kind):
                entry['status'] = 'skipped'
                print('  Existing output skipped.', flush=True)
            else:
                # Existing/incomplete outputs are deliberately not deleted or overwritten.
                log = target.with_suffix(target.suffix + '.log')
                with log.open('a') as handle:
                    handle.write('\n' + entry['time'] + '\n'); handle.flush()
                    result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
                entry.update(returncode=result.returncode, log=str(log))
                entry['status'] = 'complete' if result.returncode == 0 and complete(target, kind) else 'failed'
                if entry['status'] == 'failed':
                    errors += 1
                    print(f'  FAILED; see {log}', flush=True)
            with report.open('a') as handle:
                handle.write(json.dumps(entry) + '\n')
    print(f'Done: {len(seeds)} cases, {errors} failed operations. Report: {report}', flush=True)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Result or diagnostics directory')
    parser.add_argument('--output', type=Path, help='Default: INPUT/failure_media')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Attempt existing files too (renderers reject overwrites)')
    args = parser.parse_args()
    if args.fps <= 0 or args.speed <= 0 or args.width < 320:
        parser.error('Require fps>0, speed>0, width>=320')
    errors = run_batch(args.input, args.output or args.input / 'failure_media',
                       args.fps, args.speed, args.width, not args.no_skip_existing)
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
