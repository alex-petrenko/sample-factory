import sys

from sample_factory.enjoy import enjoy
from sf_examples.dmlab.train_dmlab import parse_dmlab_args, register_dmlab_components


def main():
    """Script entry point."""
    register_dmlab_components()
    cfg = parse_dmlab_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
