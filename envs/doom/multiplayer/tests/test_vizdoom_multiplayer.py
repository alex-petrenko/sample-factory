import time
from multiprocessing import Process
from unittest import TestCase

from envs.doom.doom_utils import make_doom_env
from envs.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnvAggregator
from envs.tests.test_envs import default_doom_cfg
from utils.utils import log, AttrDict


class TestDoom(TestCase):
    @staticmethod
    def make_standard_dm(env_config):
        cfg = default_doom_cfg()
        cfg.env_frameskip = 2
        env = make_doom_env('doom_freedm', cfg=cfg, env_config=env_config)
        env.skip_frames = cfg.env_frameskip
        return env

    @staticmethod
    def doom_multiagent(make_multi_env, worker_index, num_steps=1000):
        env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
        multi_env = make_multi_env(env_config)

        obs = multi_env.reset()

        visualize = False
        start = time.time()

        for i in range(num_steps):
            actions = [multi_env.action_space.sample()] * len(obs)
            obs, rew, dones, infos = multi_env.step(actions)

            if visualize:
                multi_env.render()

            if i % 100 == 0 or any(dones):
                log.info('Rew %r done %r info %r', rew, dones, infos)

            if all(dones):
                multi_env.reset()

        took = time.time() - start
        log.info('Took %.3f seconds for %d steps', took, num_steps)
        log.info('Server steps per second: %.1f', num_steps / took)
        log.info('Observations fps: %.1f', num_steps * multi_env.num_agents / took)
        log.info('Environment fps: %.1f', num_steps * multi_env.num_agents * multi_env.skip_frames / took)

        multi_env.close()

    def test_doom_multiagent(self):
        self.doom_multiagent(self.make_standard_dm, worker_index=0)

    def test_doom_multiagent_parallel(self):
        num_workers = 16
        workers = []

        for i in range(num_workers):
            log.info('Starting worker #%d', i)
            worker = Process(target=self.doom_multiagent, args=(self.make_standard_dm, i, 200))
            worker.start()
            workers.append(worker)
            time.sleep(0.01)

        for i in range(num_workers):
            workers[i].join()

    def test_doom_multiagent_multi_env(self):
        agents_per_env = 6
        num_envs = 2
        num_workers = 2

        skip_frames = 2  # hardcoded

        multi_env = MultiAgentEnvAggregator(
            num_envs=num_envs,
            num_workers=num_workers,
            make_env_func=self.make_standard_dm,
            stats_episodes=10,
            use_multiprocessing=True,
        )
        log.info('Before reset...')
        multi_env.reset()
        log.info('After reset...')

        actions = [multi_env.action_space.sample()] * (agents_per_env * num_envs)
        obs, rew, done, info = multi_env.step(actions)
        log.info('Rewards: %r', rew)

        start = time.time()
        num_steps = 300
        for i in range(num_steps):
            obs, rew, done, info = multi_env.step(actions)
            if i % 50 == 0:
                log.debug('Steps %d, rew: %r', i, rew)

        took = time.time() - start
        log.debug('Took %.3f sec to run %d steps, steps/sec: %.1f', took, num_steps, num_steps / took)
        log.debug('Observations fps: %.1f', num_steps * multi_env.num_agents * num_envs / took)
        log.debug('Environment fps: %.1f', num_steps * multi_env.num_agents * num_envs * skip_frames / took)

        multi_env.close()
        log.info('Done!')
