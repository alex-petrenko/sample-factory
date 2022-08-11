import sys

from sample_factory.enjoy import enjoy
from sf_examples.megaverse_examples.train_megaverse import parse_megaverse_args, register_megaverse_components


def main():
    """Script entry point."""
    register_megaverse_components()
    cfg = parse_megaverse_args(evaluation=True)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
