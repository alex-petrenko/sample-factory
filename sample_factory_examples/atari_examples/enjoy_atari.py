import sys

from sample_factory.cfg.arguments import parse_args
from sample_factory.enjoy import enjoy
from sample_factory_examples.atari_examples.train_atari import register_atari_components


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
