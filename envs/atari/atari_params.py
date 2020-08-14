def atari_override_defaults(env, parser):
    """RL params specific to Atari envs."""
    parser.set_defaults(
        encoder_subtype='convnet_simple',
        hidden_size=512,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        gamma=0.99,
        reward_clip=1.0,  # same as APE-X paper
        env_frameskip=4,
        env_framestack=4,
        exploration_loss_coeff=0.01,
    )

