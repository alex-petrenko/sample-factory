import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.atari_examples.atari_params import add_atari_env_args, atari_override_defaults
from sf_examples.atari_examples.atari_utils import ATARI_ENVS, make_atari_env


def register_atari_envs():
    for env in ATARI_ENVS:
        register_env(env.name, make_atari_env)


def register_atari_components():
    register_atari_envs()


def parse_atari_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_atari_env_args(parser)
    atari_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
