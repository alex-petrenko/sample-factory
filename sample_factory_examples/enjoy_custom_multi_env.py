import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory_examples.train_custom_multi_env import register_custom_components


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
