import argparse
import os
import subprocess
import sys
from typing import List


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convenience runner for isolated SKAB experiments.')
    parser.add_argument('--config', default='configs/config_skab.yaml', help='Path to SKAB config file')
    parser.add_argument('--output-root', default='results/experiments', help='Base output directory')
    parser.add_argument('--run-name', default=None, help='Optional fixed run name passed through to child scripts')
    parser.add_argument('--epochs', type=int, default=None, help='Optional epoch override for neural experiments')
    parser.add_argument('--seed', type=int, default=42, help='Single seed to use')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Multiple seeds for aggregation')
    parser.add_argument(
        '--mode',
        choices=['baselines', 'ablations', 'both'],
        default='both',
        help='Which SKAB experiment suites to run',
    )
    return parser.parse_args()


def _build_base_cmd(script_name: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join(project_root, 'src', 'experiments', script_name),
        '--config',
        os.path.abspath(os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config),
        '--output-root',
        args.output_root if os.path.isabs(args.output_root) else os.path.join(project_root, args.output_root),
    ]

    if args.run_name:
        cmd.extend(['--run-name', args.run_name])
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.seeds:
        cmd.extend(['--seeds', *[str(seed) for seed in args.seeds]])
    else:
        cmd.extend(['--seed', str(args.seed)])

    return cmd


def main() -> None:
    args = parse_args()

    if args.mode in ('baselines', 'both'):
        subprocess.run(_build_base_cmd('run_baselines.py', args), check=True)

    if args.mode in ('ablations', 'both'):
        subprocess.run(_build_base_cmd('run_ablations.py', args), check=True)


if __name__ == '__main__':
    main()
