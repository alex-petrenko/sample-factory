import unittest
from unittest import TestCase

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env
from envs.env_utils import quadrotors_available
from utils.timing import Timing
from utils.utils import log, is_module_available


def numba_available():
    return is_module_available('numba')


def run_multi_quadrotor_env(env_name, cfg):
    env = create_env(env_name, cfg=cfg)
    env.reset()
    for i in range(100):
        obs, r, d, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    n_frames = 1000
    env = create_env(env_name, cfg=cfg)
    env.reset()

    timing = Timing()
    with timing.timeit('step'):
        for i in range(n_frames):
            obs, r, d, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    log.debug('Time %s, FPS %.1f', timing, n_frames * env.num_agents / timing.step)
    env.close()


class TestQuads(TestCase):
    @unittest.skipUnless(quadrotors_available(), 'quadrotor env package is not installed')
    def test_quad_env(self):
        env_name = 'quadrotor_single'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))

        env = create_env(env_name, cfg=cfg)
        obs = env.reset()

        n_frames = 4000

        timing = Timing()
        with timing.timeit('step'):
            for i in range(n_frames):
                obs, r, d, info = env.step(env.action_space.sample())
                if d:
                    env.reset()

        log.debug('Time %s, FPS %.1f', timing, n_frames / timing.step)

    @unittest.skipUnless(quadrotors_available(), 'quadrotor env package is not installed')
    def test_quad_multi_env(self):
        env_name = 'quadrotor_multi'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)

    @unittest.skipUnless(quadrotors_available(), 'quadrotor env package is not installed')
    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    def test_quad_multi_env_with_numba(self):
        env_name = 'quadrotor_multi'
        cfg = default_cfg(env=env_name)
        cfg.quads_use_numba = True
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)
