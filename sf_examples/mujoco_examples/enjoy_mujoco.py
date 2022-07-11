import sys

from sample_factory.enjoy import enjoy
from sf_examples.mujoco_examples.train_mujoco import register_mujoco_components, parse_mujoco_cfg


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
