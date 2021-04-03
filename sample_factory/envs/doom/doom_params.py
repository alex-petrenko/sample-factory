from sample_factory.utils.utils import str2bool


def add_doom_env_args(env, parser):
    p = parser

    p.add_argument('--num_agents', default=-1, type=int, help='Allows to set number of agents less than number of players, to allow humans to join the match. Default value (-1) means default number defined by the environment')
    p.add_argument('--num_humans', default=0, type=int, help='Meatbags want to play?')
    p.add_argument('--num_bots', default=-1, type=int, help='Add classic (non-neural) bots to the match. If default (-1) then use number of bots specified in env cfg')
    p.add_argument('--start_bot_difficulty', default=None, type=int, help='Adjust bot difficulty, useful for evaluation')
    p.add_argument('--timelimit', default=None, type=float, help='Allows to override default match timelimit in minutes')
    p.add_argument('--res_w', default=128, type=int, help='Game frame width after resize')
    p.add_argument('--res_h', default=72, type=int, help='Game frame height after resize')
    p.add_argument('--wide_aspect_ratio', default=False, type=str2bool, help='If true render wide aspect ratio (slower but gives better FOV to the agent)')


def doom_override_defaults(env, parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        encoder_type='conv',
        encoder_subtype='convnet_simple',
        encoder_custom='vizdoom',
        hidden_size=512,
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        fps=35,
        exploration_loss='symmetric_kl',
        exploration_loss_coeff=0.001,
    )
