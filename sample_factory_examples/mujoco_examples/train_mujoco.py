import sys

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl


def register_mujoco_components():
    from sample_factory.envs.mujoco.mujoco_utils import make_mujoco_env
    from sample_factory.envs.mujoco.mujoco_params import add_mujoco_env_args, mujoco_override_defaults

    global_env_registry().register_env(
        env_name_prefix='mujoco_',
        make_env_func=make_mujoco_env,
        add_extra_params_func=add_mujoco_env_args,
        override_default_params_func=mujoco_override_defaults,
    )


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
