import sys

from sample_factory.algorithms.appo.enjoy_appo import enjoy
from sample_factory_examples.train_custom_vizdoom_env import register_custom_doom_env, custom_parse_args


def main():
    """Script entry point."""
    cfg = custom_parse_args(evaluation=True)
    register_custom_doom_env(custom_timeout=cfg.my_custom_doom_arg)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
