def env_override_defaults(env, parser):
    if env.startswith('doom'):
        from envs.doom.doom_params import doom_override_defaults
        doom_override_defaults(env, parser)
    elif env.startswith('MiniGrid'):
        from envs.minigrid.minigrid_params import minigrid_override_defaults
        minigrid_override_defaults(env, parser)
    elif env.startswith('dmlab'):
        from envs.dmlab.dmlab_params import dmlab_override_defaults
        dmlab_override_defaults(env, parser)
    elif env.startswith('atari'):
        from envs.atari.atari_params import atari_override_defaults
        atari_override_defaults(env, parser)


def add_env_args(env, parser):
    p = parser

    p.add_argument('--env_frameskip', default=None, type=int, help='Number of frames for action repeat (frame skipping). Default (None) means use default environment value')
    p.add_argument('--pixel_format', default='CHW', type=str, help='PyTorch expects CHW by default, Ray & TensorFlow expect HWC')

    if env.startswith('doom'):
        from envs.doom.doom_params import add_doom_env_args
        add_doom_env_args(env, parser)
