import copy

from envs.quadrotors.quad_multi_model import register_models
from envs.quadrotors.wrappers.additional_input import QuadsAdditionalInputWrapper
from envs.quadrotors.wrappers.discrete_actions import QuadsDiscreteActionsWrapper
from envs.quadrotors.wrappers.reward_shaping import QuadsRewardShapingWrapper, DEFAULT_QUAD_REWARD_SHAPING
from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_single(cfg, **kwargs):
    from gym_art.quadrotor_single.quadrotor import QuadrotorEnv

    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = cfg.quads_episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnv(
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, rew_coeff=rew_coeff,
    )

    if cfg.quads_discretize_actions > 0:
        env = QuadsDiscreteActionsWrapper(env, cfg.quads_discretize_actions)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)
    if cfg.quads_effort_reward is not None:
        reward_shaping['quad_rewards']['effort'] = cfg.quads_effort_reward

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping)

    if cfg.quads_clip_input:
        env = QuadsAdditionalInputWrapper(env)

    return env


def make_quadrotor_env_multi(cfg, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = cfg.quads_episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    extended_obs = cfg.neighbor_obs_type

    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, room_length=cfg.room_dims[0],
        room_width=cfg.room_dims[1], room_height=cfg.room_dims[2], rew_coeff=rew_coeff,
        quads_mode=cfg.quads_mode, quads_formation=cfg.quads_formation, quads_formation_size=cfg.quads_formation_size,
        swarm_obs=extended_obs, quads_use_numba=cfg.quads_use_numba, quads_settle=cfg.quads_settle, quads_settle_range_meters=cfg.quads_settle_range_meters,
        quads_vel_reward_out_range=cfg.quads_vel_reward_out_range, quads_obstacle_mode=cfg.quads_obstacle_mode,
        quads_view_mode=cfg.quads_view_mode, quads_obstacle_num=cfg.quads_obstacle_num, quads_obstacle_type=cfg.quads_obstacle_type, quads_obstacle_size=cfg.quads_obstacle_size,
        adaptive_env=cfg.quads_adaptive_env, obstacle_traj=cfg.quads_obstacle_traj, local_obs=cfg.quads_local_obs, obs_repr=cfg.quads_obs_repr,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius, collision_falloff_radius=cfg.quads_collision_falloff_radius,
        local_metric=cfg.quads_local_metric,
        local_coeff=cfg.quads_local_coeff,  # how much velocity matters in "distance" calculation
        use_replay_buffer=use_replay_buffer, obstacle_obs_mode=cfg.quads_obstacle_obs_mode,
        obst_penalty_fall_off=cfg.quads_obst_penalty_fall_off,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)
    if cfg.quads_effort_reward is not None:
        reward_shaping['quad_rewards']['effort'] = cfg.quads_effort_reward

    reward_shaping['quad_rewards']['quadsettle'] = cfg.quads_settle_reward
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_collision_obstacle_reward
    reward_shaping['quad_rewards']['quadcol_bin_obst_smooth_max'] = cfg.quads_collision_obst_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing)
    return env


def make_quadrotor_env(env_name, cfg=None, **kwargs):
    ensure_initialized()
    if env_name == 'quadrotor_single':
        return make_quadrotor_env_single(cfg, **kwargs)
    elif env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, **kwargs)
    else:
        raise NotImplementedError


def ensure_initialized():
    register_models()

