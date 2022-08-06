"""
Main script for training agents with SampleFactory.

"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.megaverse_examples.megaverse.megaverse_params import add_megaverse_args, megaverse_override_defaults
from sf_examples.megaverse_examples.megaverse.megaverse_utils import MEGAVERSE_ENVS, make_megaverse


def register_megaverse_envs():
    for env in MEGAVERSE_ENVS:
        register_env(env.name, make_megaverse)


def register_megaverse_components():
    register_megaverse_envs()


def parse_megaverse_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_megaverse_args(partial_cfg.env, parser)
    megaverse_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_megaverse_components()
    cfg = parse_megaverse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
