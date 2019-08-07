from unittest import TestCase

import gym
import numpy as np

from envs.doom.doom_utils import make_doom_env
from utils.utils import log


class TestMemento(TestCase):
    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_simple(env_config):
        return make_doom_env('doom_basic', memento=5)

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_advanced(env_config):
        return make_doom_env('doom_battle_hybrid', memento=5)

    def run_env(self, make_env_func):
        env = make_env_func(None)
        self.assertIsInstance(env.action_space, gym.spaces.Tuple)
        self.assertIsInstance(env.observation_space, gym.spaces.Dict)

        obs = env.reset()
        self.assertIsInstance(obs, dict)
        self.assertIn('obs', obs)
        self.assertIn('memento', obs)

        self.assertTrue(all(m == 0.0 for m in obs['memento']))

        for i in range(10):
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertIsInstance(obs, dict)
            self.assertIn('obs', obs)
            self.assertIn('memento', obs)

        log.info('Memento values: %r', obs['memento'])

    def test_basic(self):
        self.run_env(self.make_env_simple)

    def test_advanced(self):
        self.run_env(self.make_env_advanced)
