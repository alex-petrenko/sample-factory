from unittest import TestCase

import gym

from envs.doom.doom_utils import make_doom_env
from utils.utils import log, AttrDict


class TestMemento(TestCase):
    @staticmethod
    def memento_args():
        return AttrDict(dict(
            memento_size=5,
            memento_increment=0.1,
            memento_history=10,
        ))

    # noinspection PyUnusedLocal
    def make_env_simple(self, env_config):
        return make_doom_env('doom_basic', memento_args=self.memento_args())

    # noinspection PyUnusedLocal
    def make_env_advanced(self, env_config):
        return make_doom_env('doom_battle_hybrid', memento_args=self.memento_args())

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
