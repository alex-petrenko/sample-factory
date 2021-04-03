from sample_factory.envs.env_registry import global_env_registry


def env_override_defaults(env, parser):
    override_default_params_func = global_env_registry().resolve_env_name(env).override_default_params_func
    if override_default_params_func is not None:
        override_default_params_func(env, parser)


def add_env_args(env, parser):
    p = parser

    p.add_argument('--env_frameskip', default=None, type=int, help='Number of frames for action repeat (frame skipping). Default (None) means use default environment value')
    p.add_argument('--env_framestack', default=4, type=int, help='Frame stacking (only used in Atari?)')
    p.add_argument('--pixel_format', default='CHW', type=str, help='PyTorch expects CHW by default, Ray & TensorFlow expect HWC')

    add_extra_params_func = global_env_registry().resolve_env_name(env).add_extra_params_func
    if add_extra_params_func is not None:
        add_extra_params_func(env, p)
