import copy

from envs.quadrotors.wrappers.additional_input import QuadsAdditionalInputWrapper
from envs.quadrotors.wrappers.discrete_actions import QuadsDiscreteActionsWrapper
from envs.quadrotors.wrappers.reward_shaping import QuadsRewardShapingWrapper, DEFAULT_QUAD_REWARD_SHAPING


def make_quadrotor_env_single(cfg, **kwargs):
    from gym_art.quadrotor_single.quadrotor import QuadrotorEnv

    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = 7  # seconds

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

    episode_duration = 7  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnvMulti(
        num_agents=4,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, rew_coeff=rew_coeff,
    )

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)
    if cfg.quads_effort_reward is not None:
        reward_shaping['quad_rewards']['effort'] = cfg.quads_effort_reward

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping)

    return env


def make_quadrotor_env(env_name, cfg=None, **kwargs):
    if env_name == 'quadrotor_single':
        return make_quadrotor_env_single(cfg, **kwargs)
    elif env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, **kwargs)
    else:
        raise NotImplementedError()
