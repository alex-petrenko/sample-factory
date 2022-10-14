import envpool
import gym

from sample_factory.envs.env_wrappers import BatchedRecordEpisodeStatistics, EnvPoolResetFixWrapper, EnvPoolTo5Tuple

ATARI_W = ATARI_H = 84


class AtariSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


# Note NoFrameskip-v4 in gym[atari] is the same game configuration as -v5 in envpool
ATARI_ENVS = [
    AtariSpec("atari_montezuma", "MontezumaRevenge-v5", default_timeout=18000),
    AtariSpec("atari_pong", "Pong-v5"),
    AtariSpec("atari_qbert", "Qbert-v5"),
    AtariSpec("atari_breakout", "Breakout-v5"),
    AtariSpec("atari_spaceinvaders", "SpaceInvaders-v5"),
    AtariSpec("atari_asteroids", "Asteroids-v5"),
    AtariSpec("atari_gravitar", "Gravitar-v5"),
    AtariSpec("atari_mspacman", "MsPacman-v5"),
    AtariSpec("atari_seaquest", "Seaquest-v5"),
    AtariSpec("atari_beamrider", "BeamRider-v5"),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


def make_atari_env(env_name, cfg, env_config):
    assert cfg.num_envs_per_worker == 1, "when using envpool, set num_envs_per_worker=1 and use --env_agents="
    atari_spec = atari_env_by_name(env_name)

    env_kwargs = dict()

    if hasattr(cfg, "render_mode"):
        env_kwargs["render_mode"] = cfg.render_mode

    if atari_spec.default_timeout is not None:
        # envpool max_episode_steps does not take into account frameskip. see https://github.com/sail-sg/envpool/issues/195
        env_kwargs["max_episode_steps"] = atari_spec.default_timeout // 4
    if env_config is not None:
        env_kwargs["seed"] = env_config.env_id

    env = envpool.make(
        atari_spec.env_id,
        env_type="gym",
        num_envs=cfg.env_agents,
        reward_clip=True,
        episodic_life=True,
        **env_kwargs,
    )
    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=cfg.env_agents)
    env = EnvPoolTo5Tuple(env)
    env.num_agents = cfg.env_agents
    return env
