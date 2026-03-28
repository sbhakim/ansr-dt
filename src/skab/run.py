import argparse

from .pipeline import run_pipeline


DEFAULT_MODES = ['random_forest', 'neural', 'symbolic', 'neuro_symbolic']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dedicated SKAB pipeline isolated from the synthetic ANSR-DT stack.')
    parser.add_argument('--config', default='configs/config_skab_separate.yaml', help='Path to dedicated SKAB config file')
    parser.add_argument('--run-name', default=None, help='Optional fixed run directory name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=None, help='Optional neural epoch override')
    parser.add_argument('--modes', nargs='+', default=DEFAULT_MODES, choices=DEFAULT_MODES, help='Which dedicated SKAB modes to run')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        config_path=args.config,
        run_name=args.run_name,
        modes=args.modes,
        seed=args.seed,
        epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
