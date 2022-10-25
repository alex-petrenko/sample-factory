import sys

from sample_factory.enjoy import enjoy
from sf_examples.atari.train_atari import parse_atari_args, register_atari_components


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args(evaluation=True)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
