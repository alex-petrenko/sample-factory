import time
from unittest import TestCase

from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_env import MultiEnv
from envs.doom.doom_utils import make_doom_env, doom_env_by_name, make_doom_multiagent_env
from utils.timing import Timing
from utils.utils import log, AttrDict


def test_env_performance(make_env, env_type):
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
                obs, rew, done, info = env.step(env.action_space.sample())
                t.step += time.time() - start_step
                frames += num_env_steps([info])

    fps = total_num_frames / t.experience
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames on one CPU, %.1f FPS', t.experience, total_num_frames, fps)
    log.debug('Avg. reset time %.3f s', t.reset / num_resets)
    log.debug('Timing: %s', t)

    env.close()


def test_multi_env_performance(make_env, env_type, num_envs, num_workers):
    t = Timing()
    with t.timeit('init'):
        multi_env = MultiEnv(num_envs, num_workers, make_env, stats_episodes=100)
        total_num_frames, frames = 30000, 0

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
        return make_doom_env(
            doom_env_by_name('doom_battle_tuple_actions'),
        )

    @staticmethod
    def make_env_bots(env_config):
        log.info('Create host env with cfg: %r', env_config)
        return make_doom_multiagent_env(
            doom_env_by_name('doom_dwango5_bots'), env_config=env_config,
        )

    @staticmethod
    def make_env_bots_hybrid_actions(env_config):
        return make_doom_multiagent_env(
            doom_env_by_name('doom_dwango5_bots_hybrid'), env_config=env_config,
        )

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

