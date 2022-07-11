import sys

from sample_factory.cfg.arguments import parse_args
from sample_factory.enjoy import enjoy
from sf_examples.mujoco_examples.train_mujoco import register_mujoco_components


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
