# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences
import isaacgym

# isort: on

import sys

from sample_factory.enjoy import enjoy
from sf_examples.isaacgym_examples.train_isaacgym import parse_isaacgym_cfg, register_isaacgym_custom_components


def main():
    """Script entry point."""
    register_isaacgym_custom_components()
    cfg = parse_isaacgym_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
