import sys

from sample_factory.cfg.arguments import parse_args
from sample_factory.enjoy import enjoy
from sample_factory_examples.vizdoom_examples.train_vizdoom import register_vizdoom_components


def main():
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
