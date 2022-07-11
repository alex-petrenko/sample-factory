import sys

from sample_factory.enjoy import enjoy
from sf_examples.train_gym_env import parse_custom_args, register_custom_components


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
