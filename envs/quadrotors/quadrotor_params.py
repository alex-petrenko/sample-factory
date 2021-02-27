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
    p.add_argument('--quads_episode_duration', default=15.0, type=float, help='Override default value for episode duration')
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int, help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str, choices=['attention', 'mean_embed', 'mlp', 'no_encoder'], help='The type of the neighborhood encoder')

    # TODO: better default values for collision rewards
    p.add_argument('--quads_collision_reward', default=0.0, type=float, help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_obstacle_reward', default=0.0, type=float, help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor and obstacle')
    p.add_argument('--quads_settle_reward', default=0.0, type=float, help='Override default value for quadsettle reward')
    p.add_argument('--quads_settle', default=False, type=str2bool, help='Use velocity penalty and equal distance rewards when drones are within a certain radius of the goal')
    p.add_argument('--quads_vel_reward_out_range', default=0.8, type=float, help='We only use this parameter when quads_settle=True, the meaning of this parameter is that we would punish the quadrotor if it flies out of the range that we defined')
    p.add_argument('--quads_settle_range_meters', default=1.0, type=float, help='Radius of the sphere around the goal with velocity penalty to help quadrotors stop and settle at the goal')

    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float, help='When the distance between two drones are less than N arm_length, we would view them as collide.')
    p.add_argument('--quads_collision_falloff_radius', default=0.0, type=float, help='The falloff radius for the smooth penalty. 0: radius is 0 arm_length, which means we would not add extra penalty except drones collide')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')

    p.add_argument('--neighbor_obs_type', default='none', type=str, choices=['none', 'pos_vel', 'pos_vel_goals', 'pos_vel_goals_ndist_gdist'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')
    p.add_argument('--quads_obstacle_mode', default='no_obstacles', type=str, choices=['no_obstacles', 'static', 'dynamic'], help='Choose which obstacle mode to run')
    p.add_argument('--quads_obstacle_num', default=0, type=int, help='Choose the number of obstacle(s)')
    p.add_argument('--quads_obstacle_type', default='sphere', type=str, choices=['sphere', 'cube', 'random'], help='Choose the type of obstacle(s)')
    p.add_argument('--quads_obstacle_size', default=0.0, type=float, help='Choose the size of obstacle(s)')
    p.add_argument('--quads_obstacle_traj', default='gravity', type=str, choices=['gravity', 'electron'],  help='Choose the type of force to use')
    p.add_argument('--quads_local_obs', default=-1, type=int, help='Number of neighbors to consider. -1=all neighbors. 0=blind agents, 0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_local_coeff', default=0.0, type=float, help='This parameter is used for the metric of select which drones are the N closest drones.')
    p.add_argument('--quads_local_metric', default='dist_inverse', type=str, choices=['dist', 'dist_inverse'], help='The main part of evaluate the closest drones')

    p.add_argument('--quads_view_mode', default='local', type=str, choices=['local', 'global'], help='Choose which kind of view/camera to use')
    p.add_argument('--quads_adaptive_env', default=False, type=str2bool, help='Iteratively shrink the environment into a tunnel to increase obstacle density based on statistics')

    p.add_argument('--quads_mode', default='static_same_goal', type=str, choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal', 'circular_config', 'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations', 'mix', 'tunnel'], help='Choose which scenario to run. Ep = evader pursuit')
    p.add_argument('--quads_formation', default='circle_horizontal', type=str, choices=['circle_xz_vertical', 'circle_yz_vertical', 'circle_horizontal', 'sphere', 'grid_xz_vertical', 'grid_yz_vertical', 'grid_horizontal'], help='Choose the swarm formation at the goal')
    p.add_argument('--quads_formation_size', default=-1.0, type=float, help='The size of the formation, interpreted differently depending on the formation type. Default (-1) means it is determined by the mode')
    p.add_argument('--room_dims', nargs='+', default=[10, 10, 10], type=float, help='Length, width, and height dimensions respectively of the quadrotor env')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str, help='obs space for drone itself')
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float, help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this many steps. Default (0.0) is no annealing')