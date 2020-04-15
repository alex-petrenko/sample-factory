def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        hidden_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    p.add_argument('--quads_discretize_actions', default=-1, type=int, help='Discretize actions into N bins for each individual action. Default (-1) means no discretization')
