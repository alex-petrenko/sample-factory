import gym

from envs.env_wrappers import ResizeWrapper, StackFramesWrapper, SkipAndStackFramesWrapper

ATARI_W = ATARI_H = 84


class AtariCfg:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariCfg('atari_montezuma', 'MontezumaRevengeNoFrameskip-v4', default_timeout=18000),

    AtariCfg('atari_pong', 'PongNoFrameskip-v4'),
    AtariCfg('atari_breakout', 'BreakoutNoFrameskip-v4'),
    AtariCfg('atari_spaceinvaders', 'SpaceInvadersNoFrameskip-v4'),

    AtariCfg('atari_asteroids', 'AsteroidsNoFrameskip-v4'),
    AtariCfg('atari_gravitar', 'GravitarNoFrameskip-v4'),
    AtariCfg('atari_mspacman', 'MsPacmanNoFrameskip-v4'),
    AtariCfg('atari_seaquest', 'SeaQuestNoFrameskip-v4'),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Atari env')


# noinspection PyUnusedLocal
def make_atari_env(env_name, **kwargs):
    atari_cfg = atari_env_by_name(env_name)

    env = gym.make(atari_cfg.env_id)
    if atari_cfg.default_timeout is not None:
        env._max_episode_steps = atari_cfg.default_timeout

    assert 'NoFrameskip' in env.spec.id

    # if 'Montezuma' in atari_cfg.env_id or 'Pitfall' in atari_cfg.env_id:
    #     env = AtariVisitedRoomsInfoWrapper(env)

    env = ResizeWrapper(env, ATARI_W, ATARI_H, grayscale=True, add_channel_dim=False, area_interpolation=True)
    env = SkipAndStackFramesWrapper(env, skip_frames=4, stack_frames=4, channel_config='CHW')
    return env
