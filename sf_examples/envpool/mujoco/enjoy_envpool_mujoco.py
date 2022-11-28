import sys

from sample_factory.enjoy import enjoy
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components


def main():
    register_mujoco_components()
    cfg = parse_mujoco_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
