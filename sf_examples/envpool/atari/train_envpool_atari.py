import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.envpool.atari.envpool_atari_params import add_atari_env_args, atari_override_defaults
from sf_examples.envpool.atari.envpool_atari_utils import ENVPOOL_ATARI_ENVS, make_atari_env


def register_atari_envs():
    for env in ENVPOOL_ATARI_ENVS:
        register_env(env.name, make_atari_env)


def register_atari_components():
    register_atari_envs()


def parse_atari_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_atari_env_args(partial_cfg.env, parser, evaluation=evaluation)
    atari_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
