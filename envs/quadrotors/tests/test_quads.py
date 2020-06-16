import unittest
from unittest import TestCase

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import log


class TestQuads(TestCase):
    try:
        import gym_art
        gym_art_installed = True
    except ImportError:
        gym_art_installed = False

    @unittest.skipUnless(gym_art_installed, 'quadrotor env package is not installed')
    def test_quad_env(self):
        env_name = 'quadrotor_single'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))

        env = create_env(env_name, cfg=cfg)
        obs = env.reset()

        n_frames = 10000

        timing = Timing()
        with timing.timeit('step'):
            for i in range(n_frames):
                obs, r, d, info = env.step(env.action_space.sample())
                if d:
                    env.reset()

        log.debug('Time %s, FPS %.1f', timing, n_frames / timing.step)

    @unittest.skipUnless(gym_art_installed, 'quadrotor env package is not installed')
    def test_quad_multi_env(self):
        env_name = 'quadrotor_multi'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))

        env = create_env(env_name, cfg=cfg)
        env.reset()

        n_frames = 1000

        timing = Timing()
        with timing.timeit('step'):
            for i in range(n_frames):
                obs, r, d, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

        log.debug('Time %s, FPS %.1f', timing, n_frames / timing.step)
