from unittest import TestCase

from algorithms.appo import env_runner
from algorithms.appo.env_runner import VectorEnvRunner, TaskType
from algorithms.utils.algo_utils import num_env_steps
from envs.doom.doom_utils import make_doom_env
from envs.tests.test_envs import default_doom_cfg
from utils.timing import Timing
from utils.utils import log, safe_get


class TestDoom(TestCase):
    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_singleplayer(env_config):
        return make_doom_env('doom_battle_hybrid', cfg=default_doom_cfg(), env_config=env_config)

    @staticmethod
    def make_env_bots(env_config):
        log.info('Create host env with cfg: %r', env_config)
        return make_doom_env('doom_dwango5_bots_experimental', cfg=default_doom_cfg(), env_config=env_config)

    def test_single_runner(self):
        tmp_env = self.make_env_singleplayer(None)
        action_space = tmp_env.action_space
        tmp_env.close()
        del tmp_env

        # num_single_envs = 10  # FPS 2384
        # envs_single = [self.make_env_singleplayer(None) for _ in range(num_single_envs)]
        # obs = [env_single.reset() for env_single in envs_single]
        #
        # timing = Timing()
        # with timing.timeit('single_env'):
        #     num_frames = 0
        #     while num_frames < 10000:
        #         for env_single in envs_single:
        #             actions = action_space.sample()
        #             _, _, done, info = env_single.step(actions)
        #             num_frames += num_env_steps([info])
        #             if done:
        #                 env_single.reset()
        #
        # log.info('Single env fps %.1f', num_frames / timing.single_env)
        # return

        timing = Timing()

        env_runner = VectorEnvRunner(self.make_env_singleplayer, vector_size=10)  # FPS 2400
        env_runner.init()
        obs = env_runner.reset()

        # initiate the rollout
        for split in range(env_runner.num_splits):
            actions = [action_space.sample()] * len(obs[split])
            env_runner.task_queue.put((TaskType.STEP, split, actions))
            log.info('Initialized rollout for split %d', split)

        num_frames, max_num_frames = 0, 10000

        with timing.timeit('runner'):
            while num_frames < max_num_frames:
                # log.info('Waiting for result...')
                results = safe_get(env_runner.result_queue)
                split, results = results

                env_runner.result_queue.task_done()

                # log.info('Calculating actions for split %d', split)
                actions = [action_space.sample()] * len(obs[split])
                env_runner.task_queue.put((TaskType.STEP, split, actions))

                num_frames += num_env_steps([r[-1] for r in results])  # parse infos
                pass

        log.info('FPS is %.1f', num_frames / timing.runner)
        env_runner.close()
