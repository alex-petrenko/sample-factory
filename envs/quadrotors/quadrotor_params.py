def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='fc',
        encoder='fc',
        hidden_size=256,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser
