import sys

from sample_factory.enjoy import enjoy
from sf_examples.brax.train_brax import parse_brax_cfg, register_brax_custom_components


def main():
    """Script entry point."""
    register_brax_custom_components(evaluation=True)
    cfg = parse_brax_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
