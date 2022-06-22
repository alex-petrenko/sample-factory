import sys

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl


def register_vizdoom_components():
    from sample_factory.envs.doom.doom_utils import make_doom_env
    from sample_factory.envs.doom.doom_params import add_doom_env_args, doom_override_defaults

    global_env_registry().register_env(
        env_name_prefix='doom_',
        make_env_func=make_doom_env,
        add_extra_params_func=add_doom_env_args,
        override_default_params_func=doom_override_defaults,
    )

    from sample_factory.envs.doom.doom_model import register_models
    register_models()


def main():
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
