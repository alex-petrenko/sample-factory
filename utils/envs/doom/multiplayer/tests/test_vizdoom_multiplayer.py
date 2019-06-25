import time
from multiprocessing import Process
from unittest import TestCase

import cv2

from utils.envs.doom.doom_helpers import concat_grid
from utils.envs.doom.doom_utils import doom_env_by_name, make_doom_env
from utils.envs.doom.multiplayer.doom_multiagent import VizdoomMultiAgentEnv
from utils.utils import log


class TestDoom(TestCase):
    @staticmethod
    def make_env_dm(player_id, num_players):
        return make_doom_env(
            doom_env_by_name('doom_dm'), player_id=player_id, num_players=num_players, skip_frames=True,
        )

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env_dm(player_id=0, num_players=8))

    def doom_multiagent(self, worker_index):
        env_config = {'worker_index': worker_index, 'vector_index': 0}
        num_players = 8
        multi_env = VizdoomMultiAgentEnv(num_players=num_players, make_env_func=self.make_env_dm, env_config=env_config)

        obs = multi_env.reset()

        num_steps = 5000
        visualize = False
        start = time.time()

        for i in range(num_steps):
            actions = {key: multi_env.action_space.sample() for key in obs.keys()}
            obs, rew, dones, infos = multi_env.step(actions)

            if visualize:
                obs_display = [o['obs'] for o in obs.values()]
                obs_grid = concat_grid(obs_display)
                cv2.imshow('vizdoom', obs_grid)
                cv2.waitKey(1)

            if i % 100 == 0 or any(dones.values()):
                log.info('Rew %r done %r info %r', rew, dones, infos)

            if dones['__all__']:
                multi_env.reset()

        took = time.time() - start
        log.info('Took %.3f seconds for %d steps', took, num_steps)
        log.info('Server steps per second: %.1f', num_steps / took)
        log.info('Observations fps: %.1f', num_steps * num_players / took)

        skip_frames = 4
        log.info('Environment fps: %.1f', num_steps * num_players * skip_frames / took)

        multi_env.close()

    def test_doom_multiagent(self):
        self.doom_multiagent(worker_index=0)

    def test_doom_multiagent_parallel(self):
        num_workers = 16
        workers = []

        for i in range(num_workers):
            log.info('Starting worker #%d', i)
            worker = Process(target=self.doom_multiagent, args=(i,))
            worker.start()
            workers.append(worker)

        for i in range(num_workers):
            workers[i].join()
