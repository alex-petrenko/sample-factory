# this is here just to guarantee that isaacgym is imported before PyTorch
# noinspection PyUnresolvedReferences
import isaacgym

import sys

from sample_factory.cfg.arguments import parse_args
from sample_factory.enjoy import enjoy
from sample_factory_examples.isaacgym_examples.train_isaacgym import register_isaacgym_custom_components


def main():
    """Script entry point."""
    register_isaacgym_custom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
