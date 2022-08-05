import gym

from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
)

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariSpec("atari_montezuma", "MontezumaRevengeNoFrameskip-v4", default_timeout=18000),
    AtariSpec("atari_pong", "PongNoFrameskip-v4"),
    AtariSpec("atari_qbert", "QbertNoFrameskip-v4"),
    AtariSpec("atari_breakout", "BreakoutNoFrameskip-v4"),
    AtariSpec("atari_spaceinvaders", "SpaceInvadersNoFrameskip-v4"),
    AtariSpec("atari_asteroids", "AsteroidsNoFrameskip-v4"),
    AtariSpec("atari_gravitar", "GravitarNoFrameskip-v4"),
    AtariSpec("atari_mspacman", "MsPacmanNoFrameskip-v4"),
    AtariSpec("atari_seaquest", "SeaquestNoFrameskip-v4"),
    AtariSpec("atari_beamrider", "BeamRiderNoFrameskip-v4"),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


# noinspection PyUnusedLocal
def make_atari_env(env_name, cfg, env_config):
    atari_spec = atari_env_by_name(env_name)

    env_kwargs = dict()
    if hasattr(cfg, "render_mode"):
        env_kwargs["render_mode"] = cfg.render_mode
    env = gym.make(atari_spec.env_id, **env_kwargs)

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
