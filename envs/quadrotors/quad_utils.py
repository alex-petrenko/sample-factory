from gym_art.quadrotor.quadrotor import QuadrotorEnv

from envs.quadrotors.wrappers.discrete_actions import QuadsDiscreteActionsWrapper
from envs.quadrotors.wrappers.reward_shaping import QuadsRewardShapingWrapper, DEFAULT_QUAD_REWARD_SHAPING


def make_quadrotor_env(env_name, cfg=None, **kwargs):
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

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=DEFAULT_QUAD_REWARD_SHAPING)

    return env
