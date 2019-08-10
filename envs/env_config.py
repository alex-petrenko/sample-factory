from utils.utils import log


def env_override_default_params_and_args(env_name, params, args):
    """Override default algorithm params based on env type."""
    if env_name.startswith('doom'):
        from envs.doom.doom_params import doom_override_default_params_and_args
        doom_override_default_params_and_args(params, args)
    elif env_name.startswith('MiniGrid'):
        from envs.minigrid.minigrid_params import minigrid_override_default_params_and_args
        minigrid_override_default_params_and_args(params, args)
    else:
        log.warning('Environment %r does not override default params')


def add_env_args(parser):
    parser.add_argument(
        '--num_agents',
        default=-1,
        type=int,
        help='Allows to set number of agents less than number of players, to allow humans to join the match'
             'Default value (-1) means number of agents is the same as max number of players',
    )
    parser.add_argument(
        '--num_humans',
        default=0,
        type=int,
        help='Meatbags want to play?',
    )
    parser.add_argument(
        '--num_bots',
        default=-1,
        type=int,
        help='Add classic (non-neural) bots to the match. If default (-1) then use number of bots specified in env cfg',
    )
    parser.add_argument(
        '--bot_difficulty',
        default=150,
        type=int,
        help='Adjust bot difficulty',
    )
    parser.add_argument(
        '--env_frameskip',
        default=None,
        type=int,
        help='Environment frameskip, None means default',
    )
    parser.add_argument(
        '--pixel_format',
        default='CHW',
        type=str,
        help='PyTorch expects CHW by default, Ray & TensorFlow expect HWC',
    )

