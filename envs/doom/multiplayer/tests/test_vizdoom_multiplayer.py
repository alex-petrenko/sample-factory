import time
from multiprocessing import Process
from unittest import TestCase

from envs.doom.doom_utils import make_doom_multiplayer_env, doom_env_by_name
from envs.tests.test_envs import default_doom_cfg
from utils.utils import log, AttrDict


class TestDoom(TestCase):
    @staticmethod
    def make_standard_dm(env_config):
        return make_doom_multiplayer_env(
            doom_env_by_name('doom_dm'), cfg=default_doom_cfg(), env_config=env_config,
        )

    @staticmethod
    def doom_multiagent(make_multi_env, worker_index, num_steps=1000):
        env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0, 'safe_init': False})
        multi_env = make_multi_env(env_config)

        obs = multi_env.reset()

        visualize = False
        start = time.time()

        for i in range(num_steps):
            actions = {key: multi_env.action_space.sample() for key in obs.keys()}
            obs, rew, dones, infos = multi_env.step(actions)

            if visualize:
                multi_env.render()

            if i % 100 == 0 or any(dones.values()):
                log.info('Rew %r done %r info %r', rew, dones, infos)

            if dones['__all__']:
                multi_env.reset()

        took = time.time() - start
        log.info('Took %.3f seconds for %d steps', took, num_steps)
        log.info('Server steps per second: %.1f', num_steps / took)
        log.info('Observations fps: %.1f', num_steps * multi_env.num_agents / took)
        log.info('Environment fps: %.1f', num_steps * multi_env.num_agents * multi_env.skip_frames / took)

        multi_env.close()

    def test_doom_multiagent(self):
        self.doom_multiagent(self.make_standard_dm, worker_index=0)

    # fails often when one or two clients can't connect (TODO!)
    def skipped_test_doom_multiagent_parallel(self):
        num_workers = 16
        workers = []

        for i in range(num_workers):
            log.info('Starting worker #%d', i)
            worker = Process(target=self.doom_multiagent, args=(self.make_standard_dm, i, 200))
            worker.start()
            workers.append(worker)

        for i in range(num_workers):
            workers[i].join()
