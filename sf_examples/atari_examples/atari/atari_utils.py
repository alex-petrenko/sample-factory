from typing import Optional

import gym

from sample_factory.envs.env_wrappers import (
    BatchedRecordEpisodeStatistics,
    ClipRewardEnv,
    EnvPoolResetFixWrapper,
    EnvPoolTo5Tuple,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
)

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, envpool_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.envpool_id = envpool_id
        self.default_timeout = default_timeout
        self.has_timer = False


# Note NoFrameskip-v4 in gym[atari] is the same game configuration as -v5 in envpool
ATARI_ENVS = [
    AtariSpec("atari_montezuma", "MontezumaRevengeNoFrameskip-v4", "MontezumaRevenge-v5", default_timeout=18000),
    AtariSpec("atari_pong", "PongNoFrameskip-v4", "Pong-v5"),
    AtariSpec(
        "atari_qbert",
        "QbertNoFrameskip-v4",
        "Qbert-v5",
    ),
    AtariSpec("atari_breakout", "BreakoutNoFrameskip-v4", "Breakout-v5"),
    AtariSpec("atari_spaceinvaders", "SpaceInvadersNoFrameskip-v4", "SpaceInvaders-v5"),
    AtariSpec("atari_asteroids", "AsteroidsNoFrameskip-v4", "Asteroids-v5"),
    AtariSpec(
        "atari_gravitar",
        "GravitarNoFrameskip-v4",
        "Gravitar-v5",
    ),
    AtariSpec("atari_mspacman", "MsPacmanNoFrameskip-v4", "MsPacman-v5"),
    AtariSpec(
        "atari_seaquest",
        "SeaquestNoFrameskip-v4",
        "Seaquest-v5",
    ),
    AtariSpec("atari_beamrider", "BeamRiderNoFrameskip-v4", "BeamRider-v5"),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


def make_atari_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    atari_spec = atari_env_by_name(env_name)
    env = envpool.make(
        atari_spec.envpool_id,
        env_type="gym",
        num_envs=cfg.env_agents,
        reward_clip=True,
        episodic_life=True,
        max_episode_steps=18000,
    )  # TODO set max_episode steps from spec
    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=cfg.env_agents)
    env = EnvPoolTo5Tuple(env)
    env.num_agents = cfg.env_agents
    return env

    env = gym.make(atari_spec.env_id, render_mode=render_mode)

    if atari_spec.default_timeout is not None:
        env._max_episode_steps = atari_spec.default_timeout

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = EpisodicLifeEnv(env)
    # noinspection PyUnresolvedReferences
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)
    return env
