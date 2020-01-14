import random

from ray.tune import register_env

from algorithms.utils.arguments import default_cfg
from envs.doom.doom_utils import DOOM_ENVS, make_doom_env


def register_doom_envs_rllib(**kwargs):
    """Register env factories in RLLib system."""
    for spec in DOOM_ENVS:
        def make_env_func(env_config):
            print('Creating env!!!')
            cfg = default_cfg(env=spec.name)
            cfg.pixel_format = 'HWC'  # tensorflow models expect HWC by default

            if 'skip_frames' in env_config:
                cfg.env_frameskip = env_config['skip_frames']
            if 'res_w' in env_config:
                cfg.res_w = env_config['res_w']
            if 'res_h' in env_config:
                cfg.res_h = env_config['res_h']
            if 'wide_aspect_ratio' in env_config:
                cfg.wide_aspect_ratio = env_config['wide_aspect_ratio']

            env = make_doom_env(spec.name, env_config=env_config, cfg=cfg, **kwargs)

            import time
            time.sleep(random.random() * 5)
            print('Env created!!!')
            env.reset()

            return env

        register_env(spec.name, make_env_func)
