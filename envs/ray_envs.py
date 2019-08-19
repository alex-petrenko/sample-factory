from ray.tune import register_env

from algorithms.utils.arguments import default_cfg
from envs.doom.doom_utils import DOOM_ENVS, make_doom_env


def register_doom_envs_rllib(**kwargs):
    """Register env factories in RLLib system."""
    for spec in DOOM_ENVS:
        def make_env_func(env_config):
            cfg = default_cfg(env=spec.name)
            cfg.pixel_format = 'HWC'  # tensorflow models expect HWC by default

            if 'skip_frames' in env_config:
                cfg.env_frameskip = env_config['skip_frames']

            env = make_doom_env(spec.name, env_config=env_config, cfg=cfg, **kwargs)
            return env

        register_env(spec.name, make_env_func)
