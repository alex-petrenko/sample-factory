from ray.tune import register_env

from envs.doom.doom_utils import DOOM_ENVS, make_doom_env


def register_doom_envs_rllib(**kwargs):
    """Register env factories in RLLib system."""
    for cfg in DOOM_ENVS:
        register_env(
            cfg.name,
            lambda config: make_doom_env(cfg.name, env_config=config, **kwargs),
        )
