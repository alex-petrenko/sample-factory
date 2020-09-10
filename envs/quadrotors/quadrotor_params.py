from utils.utils import str2bool


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
    p.add_argument('--quads_clip_input', default=False, type=str2bool, help='Whether to clip input to ensure it stays relatively small')
    p.add_argument('--quads_effort_reward', default=None, type=float, help='Override default value for effort reward')
    p.add_argument('--quads_episode_duration', default=7.0, type=float, help='Override default value for episode duration')
    p.add_argument('--quads_num_agents', default=4, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_collision_reward', default=None, type=float, help='Override default value for quadcol_bin reward')
    p.add_argument('--quads_settle_reward', default=None, type=float, help='Override default value for quadsettle reward')
    p.add_argument('--quads_dist_between_goals', default=0.3, type=float, help='Under circular configuration scenarios, it should be the radius of the circle of goals')
    p.add_argument('--quads_mode', default='circular_config', type=str, choices=['circular_config', 'same_goal'], help='Choose which scanerio to run')
