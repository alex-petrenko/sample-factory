import unittest
from unittest import TestCase

from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.create_env import create_env
from sample_factory.envs.env_utils import minigrid_available
from sample_factory.utils.utils import log


class TestMinigrid(TestCase):
    @unittest.skipUnless(minigrid_available(), 'Gym minigrid not installed')
    def test_minigrid_env(self):
        env_name = 'MiniGrid-Empty-Random-5x5-v0'
        env = create_env(env_name, cfg=default_cfg(env=env_name))
        log.info('Env action space: %r', env.action_space)
        log.info('Env obs space: %r', env.observation_space)

        env.reset()
        total_rew = 0
        for i in range(1000):
            obs, rew, done, info = env.step(env.action_space.sample())
            total_rew += rew
            if done:
                env.reset()

