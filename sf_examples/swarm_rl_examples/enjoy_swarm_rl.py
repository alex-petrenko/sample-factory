import sys

from sample_factory.enjoy import enjoy
from sf_examples.swarm_rl_examples.train_swarm_rl import parse_swarm_cfg, register_swarm_components


def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
