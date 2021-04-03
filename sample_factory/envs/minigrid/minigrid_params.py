# noinspection PyUnusedLocal
def minigrid_override_defaults(env, parser):
    parser.set_defaults(
        encoder_subtype='minigrid_convnet_tiny',
        hidden_size=128,
        obs_subtract_mean=4.0,
        obs_scale=8.0,
        exploration_loss_coeff=0.005,
        env_frameskip=1,
    )
