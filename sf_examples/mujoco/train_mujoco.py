import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.mujoco.mujoco_params import add_mujoco_env_args, mujoco_override_defaults
from sf_examples.mujoco.mujoco_utils import MUJOCO_ENVS, make_mujoco_env


def register_mujoco_components():
    for env in MUJOCO_ENVS:
        register_env(env.name, make_mujoco_env)


def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_mujoco_env_args(partial_cfg.env, parser)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
