from argparse import ArgumentParser

from sample_factory.algo.utils.context import global_env_registry


def env_override_defaults(env, parser):
    override_default_params_func = global_env_registry().resolve_env_name(env).override_default_params_func
    if override_default_params_func is not None:
        override_default_params_func(env, parser)


def add_env_args(env, parser: ArgumentParser):
    add_extra_params_func = global_env_registry().resolve_env_name(env).add_extra_params_func
    if add_extra_params_func is not None:
        add_extra_params_func(env, parser)
