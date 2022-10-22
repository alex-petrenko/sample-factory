from typing import Optional

try:
    import envpool
except ImportError as e:
    print(e)
    print("Trying to import envpool when it is not installed. install with 'pip install envpool'")

from sf_examples.atari.atari_utils import ATARI_ENVS, AtariSpec
from sf_examples.envpool.envpool_wrappers import BatchedRecordEpisodeStatistics, EnvPoolResetFixWrapper

# Note NoFrameskip-v4 in gym[atari] is the same game configuration as -v5 in envpool
ENVPOOL_ATARI_ENVS = [
    AtariSpec(
        spec.name,
        spec.env_id.replace("NoFrameskip-v4", "-v5"),
        default_timeout=spec.default_timeout,
    )
    for spec in ATARI_ENVS
]


def atari_env_by_name(name):
    for cfg in ENVPOOL_ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Atari env")


def make_atari_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    assert cfg.num_envs_per_worker == 1, "when using envpool, set num_envs_per_worker=1 and use --env_agents="
    atari_spec = atari_env_by_name(env_name)

    env_kwargs = dict()

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
        frame_skip=cfg.env_frameskip,
        **env_kwargs,
    )
    env = EnvPoolResetFixWrapper(env)
    env = BatchedRecordEpisodeStatistics(env, num_envs=cfg.env_agents)
    env.num_agents = cfg.env_agents
    return env
