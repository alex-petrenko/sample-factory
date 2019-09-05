def atari_override_defaults(env, parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        encoder='convnet_simple',
        hidden_size=512,
        obs_subtract_mean=128.0,
        obs_scale=128.0,
        gamma=0.997,  # same as R2D2 paper
        reward_clip=1.0,  # same as APE-X paper
        env_frameskip=4,
        prior_loss_coeff=0.01,
    )
