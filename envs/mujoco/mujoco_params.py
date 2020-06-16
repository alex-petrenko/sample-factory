def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_mujoco',
        hidden_size=64,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
        nonlinearity='tanh'
    )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    p = parser
