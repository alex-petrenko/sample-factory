import gym

from sample_factory.envs.env_wrappers import ResizeWrapper, SkipAndStackFramesWrapper, SkipFramesWrapper, \
    PixelFormatChwWrapper

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariSpec('atari_montezuma', 'MontezumaRevengeNoFrameskip-v4', default_timeout=18000),

    AtariSpec('atari_pong', 'PongNoFrameskip-v4'),
    AtariSpec('atari_qbert', 'QbertNoFrameskip-v4'),
    AtariSpec('atari_breakout', 'BreakoutNoFrameskip-v4'),
    AtariSpec('atari_spaceinvaders', 'SpaceInvadersNoFrameskip-v4'),

    AtariSpec('atari_asteroids', 'AsteroidsNoFrameskip-v4'),
    AtariSpec('atari_gravitar', 'GravitarNoFrameskip-v4'),
    AtariSpec('atari_mspacman', 'MsPacmanNoFrameskip-v4'),
    AtariSpec('atari_seaquest', 'SeaQuestNoFrameskip-v4'),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Atari env')


# noinspection PyUnusedLocal
def make_atari_env(env_name, cfg, **kwargs):
    atari_spec = atari_env_by_name(env_name)

    env = gym.make(atari_spec.env_id)
    if atari_spec.default_timeout is not None:
        env._max_episode_steps = atari_spec.default_timeout

    assert 'NoFrameskip' in env.spec.id

    # if 'Montezuma' in atari_cfg.env_id or 'Pitfall' in atari_cfg.env_id:
    #     env = AtariVisitedRoomsInfoWrapper(env)

    add_channel_dim = cfg.env_framestack == 1
    env = ResizeWrapper(
        env, ATARI_W, ATARI_H, grayscale=True, add_channel_dim=add_channel_dim, area_interpolation=False,
    )

    pixel_format = cfg.pixel_format if 'pixel_format' in cfg else 'HWC'
    if pixel_format == 'CHW' and add_channel_dim:
        env = PixelFormatChwWrapper(env)

    if cfg.env_framestack == 1:
        env = SkipFramesWrapper(env, skip_frames=cfg.env_frameskip)
    else:
        env = SkipAndStackFramesWrapper(env, skip_frames=cfg.env_frameskip, stack_frames=4, channel_config='CHW')
    return env
