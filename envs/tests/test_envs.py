import os
import time
from os.path import join
from unittest import TestCase

from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.arguments import default_cfg
from algorithms.utils.multi_env import MultiEnv
from envs.doom.doom_gym import VizdoomEnv
from envs.doom.doom_utils import make_doom_env
from utils.timing import Timing
from utils.utils import log, AttrDict


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


def test_multi_env_performance(make_env, env_type, num_envs, num_workers, total_num_frames=30000):
    t = Timing()
    frames = 0

    with t.timeit('init'):
        multi_env = MultiEnv(num_envs, num_workers, make_env, stats_episodes=100)

    with t.timeit('first_reset'):
        multi_env.reset()

    next_print = print_step = 10000

    with t.timeit('experience'):
        while frames < total_num_frames:
            _, rew, done, info = multi_env.step([multi_env.action_space.sample()] * num_envs)
            frames += num_env_steps(info)
            if frames > next_print:
                log.info('Collected %d frames of experience...', frames)
                next_print += print_step

    fps = total_num_frames / t.experience
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames in parallel, %.1f FPS', t.experience, total_num_frames, fps)
    log.debug('Timing: %s', t)

    multi_env.close()


class TestDoom(TestCase):
    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_singleplayer(env_config):
        return make_doom_env('doom_battle_hybrid', cfg=default_doom_cfg(), env_config=env_config)

    @staticmethod
    def make_env_bots(env_config):
        log.info('Create host env with cfg: %r', env_config)
        return make_doom_env('doom_dwango5_bots', cfg=default_doom_cfg(), env_config=env_config)

    @staticmethod
    def make_env_bots_hybrid_actions(env_config, **kwargs):
        return make_doom_env('doom_dwango5_bots_hybrid', cfg=default_doom_cfg(), env_config=env_config, **kwargs)

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env_singleplayer(None))

    def test_doom_performance(self):
        test_env_performance(self.make_env_singleplayer, 'doom')

    def test_doom_performance_multi(self):
        test_multi_env_performance(self.make_env_singleplayer, 'doom', num_envs=200, num_workers=20)

    def test_doom_performance_bots(self):
        test_env_performance(self.make_env_bots, 'doom')

    def test_doom_performance_bots_hybrid_actions(self):
        test_env_performance(self.make_env_bots_hybrid_actions, 'doom')

    def test_doom_performance_bots_multi(self):
        test_multi_env_performance(self.make_env_bots, 'doom', num_envs=200, num_workers=20)

    def test_doom_two_color(self):
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

        demo_path = join(rec_dir, VizdoomEnv.demo_name(episode_idx=0))

        env = self.make_env_bots_hybrid_actions(None, custom_resolution='1920x1080')

        VizdoomEnv.replay(env, demo_path)

        self.assertTrue(os.path.isfile(demo_path))
        os.remove(demo_path)
        self.assertFalse(os.path.isfile(demo_path))


class TestDmlab(TestCase):
    """DMLab tests fail too often just randomly (EGL errors), so we're skipping them for now."""

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        from envs.dmlab.dmlab_utils import make_dmlab_env
        return make_dmlab_env('dmlab_nonmatch', cfg=default_cfg(env='dmlab_nonmatch'))

    def test_dmlab_performance(self):
        test_env_performance(self.make_env, 'dmlab')

    def test_dmlab_performance_multi(self):
        test_multi_env_performance(self.make_env, 'dmlab', num_envs=64, num_workers=64, total_num_frames=int(3e5))
