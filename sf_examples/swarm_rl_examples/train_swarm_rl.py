"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.swarm_rl_examples.swarm_rl.quad_multi_model import register_models
from sf_examples.swarm_rl_examples.swarm_rl.quad_utils import make_quadrotor_env
from sf_examples.swarm_rl_examples.swarm_rl.quadrotor_params import (
    add_quadrotors_env_args,
    quadrotors_override_defaults,
)


def register_swarm_components():
    for env in ["quadrotor_single", "quadrotor_multi"]:
        register_env(env, make_quadrotor_env)

    register_models()


def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=False)
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
