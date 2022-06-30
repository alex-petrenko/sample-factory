import sys

from sample_factory.cfg.arguments import parse_args
from sample_factory.enjoy import enjoy

from sample_factory_examples.swarm_rl_examples.train_swarm_rl import register_custom_components


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
