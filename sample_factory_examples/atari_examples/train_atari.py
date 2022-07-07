import sys

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl


def register_atari_components():
    from sample_factory.envs.atari.atari_params import add_atari_env_args, atari_override_defaults
    from sample_factory.envs.atari.atari_utils import make_atari_env

    global_env_registry().register_env(
        env_name_prefix="atari_",
        make_env_func=make_atari_env,
        add_extra_params_func=add_atari_env_args,
        override_default_params_func=atari_override_defaults,
    )


def main():
    """Script entry point."""
    register_atari_components()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
