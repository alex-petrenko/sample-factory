from filelock import FileLock, Timeout
from ray.tune import register_env

from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.dmlab.dmlab_env import DMLAB_ENVS, make_dmlab_env
from sample_factory.envs.doom.doom_utils import DOOM_ENVS, make_doom_env

DOOM_LOCK_PATH = '/tmp/doom_rllib_lock'


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

            # we lock the global mutex here, otherwise Doom instances may crash on first reset when too many of them are reset simultaneously
            lock = FileLock(DOOM_LOCK_PATH)
            attempt = 0
            while True:
                attempt += 1
                try:
                    with lock.acquire(timeout=10):
                        print('Env created, resetting...')
                        env.reset()
                        print('Env reset completed! Config:', env_config)
                        break
                except Timeout:
                    print('Another instance of this application currently holds the lock, attempt:', attempt)

            return env

        register_env(spec.name, make_env_func)


def register_dmlab_envs_rllib(**kwargs):
    for spec in DMLAB_ENVS:
        def make_env_func(env_config):
            print('Creating env!!!')
            cfg = default_cfg(env=spec.name)
            cfg.pixel_format = 'HWC'  # tensorflow models expect HWC by default

            if 'res_w' in env_config:
                cfg.res_w = env_config['res_w']
            if 'res_h' in env_config:
                cfg.res_h = env_config['res_h']
            if 'renderer' in env_config:
                cfg.renderer = env_config['renderer']
            if 'dmlab_throughput_benchmark' in env_config:
                cfg.renderer = env_config['dmlab_throughput_benchmark']

            env = make_dmlab_env(spec.name, env_config=env_config, cfg=cfg, **kwargs)
            return env

        register_env(spec.name, make_env_func)
