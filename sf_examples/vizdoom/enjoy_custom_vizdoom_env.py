import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults
from sf_examples.vizdoom.train_custom_vizdoom_env import add_custom_args, register_custom_doom_env
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components


def main():
    """Script entry point."""
    register_vizdoom_components()
    parser, cfg = parse_sf_args(evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    doom_override_defaults(parser)
    add_custom_args(parser)
    cfg = parse_full_cfg(parser)

    register_custom_doom_env(custom_timeout=cfg.my_custom_doom_arg)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
