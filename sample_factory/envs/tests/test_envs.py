import os
import time
import unittest
from os.path import join
from unittest import TestCase

from sample_factory.algorithms.utils.algo_utils import num_env_steps
from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.dmlab.dmlab_utils import string_to_hash_bucket

from sample_factory.envs.env_utils import dmlab_available, vizdoom_available
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict


def default_doom_cfg():
    return default_cfg(env='doom_env')


def test_env_performance(make_env, env_type, verbose=False):
    t = Timing()
    with t.timeit('init'):
        env = make_env(AttrDict({'worker_index': 0, 'vector_index': 0}))
        total_num_frames, frames = 10000, 0

    with t.timeit('first_reset'):
        env.reset()

    t.reset = t.step = 1e-9
    num_resets = 0
    with t.timeit('experience'):
        while frames < total_num_frames:
            done = False

            start_reset = time.time()
            env.reset()

            t.reset += time.time() - start_reset
            num_resets += 1

            while not done and frames < total_num_frames:
                start_step = time.time()
                if verbose:
                    env.render()
                    time.sleep(1.0 / 40)

                obs, rew, done, info = env.step(env.action_space.sample())
                if verbose:
                    log.info('Received reward %.3f', rew)

                t.step += time.time() - start_step
                frames += num_env_steps([info])

    fps = total_num_frames / t.experience
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames on one CPU, %.1f FPS', t.experience, total_num_frames, fps)
    log.debug('Avg. reset time %.3f s', t.reset / num_resets)
    log.debug('Timing: %s', t)
    env.close()


@unittest.skipUnless(vizdoom_available(), 'Please install VizDoom to run a full test suite')
class TestDoom(TestCase):
    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_singleplayer(env_config):
        from sample_factory.envs.doom.doom_utils import make_doom_env
        return make_doom_env('doom_benchmark', cfg=default_doom_cfg(), env_config=env_config)

    @staticmethod
    def make_env_bots_hybrid_actions(env_config, **kwargs):
        from sample_factory.envs.doom.doom_utils import make_doom_env
        return make_doom_env('doom_deathmatch_bots', cfg=default_doom_cfg(), env_config=env_config, **kwargs)

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env_singleplayer(None))

    def test_doom_performance(self):
        test_env_performance(self.make_env_singleplayer, 'doom')

    # def test_doom_performance_multi(self):
    #     test_multi_env_performance(self.make_env_singleplayer, 'doom', num_envs=2, num_workers=2)

    def test_doom_performance_bots_hybrid_actions(self):
        test_env_performance(self.make_env_bots_hybrid_actions, 'doom')

    # def test_doom_performance_bots_multi(self):
    #     test_multi_env_performance(self.make_env_bots_hybrid_actions, 'doom', num_envs=200, num_workers=20)

    def test_doom_two_color(self):
        from sample_factory.envs.doom.doom_utils import make_doom_env
        test_env_performance(
            lambda env_config: make_doom_env('doom_two_colors_easy', cfg=default_doom_cfg()), 'doom', verbose=False,
        )

    def skip_test_recording(self):
        # this seems to be broken in the last version of VizDoom
        rec_dir = '/tmp/'
        env = self.make_env_bots_hybrid_actions(None, record_to=rec_dir)
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())

        env.reset()
        env.close()

        from sample_factory.envs.doom.doom_gym import VizdoomEnv
        demo_path = join(rec_dir, VizdoomEnv.demo_path(episode_idx=0))

        env = self.make_env_bots_hybrid_actions(None, custom_resolution='1920x1080')

        VizdoomEnv.replay(env, demo_path)

        self.assertTrue(os.path.isfile(demo_path))
        os.remove(demo_path)
        self.assertFalse(os.path.isfile(demo_path))


class TestAtari(TestCase):
    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        from sample_factory.envs.atari.atari_utils import make_atari_env
        return make_atari_env('atari_breakout', cfg=default_cfg(env='atari_breakout'))

    def test_atari_performance(self):
        test_env_performance(self.make_env, 'atari')


class TestDmlab(TestCase):
    """DMLab tests fail too often just randomly (EGL errors), so we're skipping them for now."""

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        from sample_factory.envs.dmlab.dmlab_env import make_dmlab_env
        return make_dmlab_env('dmlab_nonmatch', cfg=default_cfg(env='dmlab_nonmatch'), env_config=None)

    @unittest.skipUnless(dmlab_available(), 'Dmlab package not installed')
    def test_dmlab_performance(self):
        test_env_performance(self.make_env, 'dmlab')

    def test_hash_bucket(self):
        vocab_size = 42
        data = {
            'a cupful of liquid that was almost, but not quite, entirely unlike tea': 37,
            'aaabbbccc': 28,
            '12313123132': 28,
            'RL': 24,
            'dmlab': 35,
        }
        for s, h in data.items():
            self.assertEqual(string_to_hash_bucket(s, vocab_size), h)
