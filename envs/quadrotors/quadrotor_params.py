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
    p.add_argument('--quads_dist_between_goals', default=0.0, type=float, help='Under circular configuration scenarios, it should be the radius of the circle of goals')
    p.add_argument('--quads_mode', default='static_goal', type=str, choices=['static_goal', 'dynamic_goal', 'circular_config'], help='Choose which scanerio to run')
    p.add_argument('--extend_obs', default=False, type=str2bool, help='Drones receive relative pos and relative vel info from all other drones')
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')
    p.add_argument('--quads_settle', default=False, type=str2bool, help='Use velocity penalty and equal distance rewards when drones are within a certain radius of the goal')
    p.add_argument('--quads_settle_range_coeff', default=10, type=float, help='The coefficient of the range that we hope a quadrotor settled in, range = quads_settle_range_coeff * the arm length of drone')
    p.add_argument('--quads_vel_reward_out_range', default=0.8, type=float, help='We only use this parameter when quads_settle=True, the meaning of this parameter is that we would punish the quadrotor if it flies out of the range that we defined')
    p.add_argument('--quads_goal_dimension', default='2D', type=str, choices=['2D', '3D'], help='Choose which dimension of goal to use')
    p.add_argument('--quads_obstacle_mode', default='None', type=str, choices=['None', 'static', 'dynamic'], help='Choose which obstacle mode to run')
    p.add_argument('--quads_view_mode', default='local', type=str, choices=['local', 'global'], help='Choose which kind of view/camera to use')
