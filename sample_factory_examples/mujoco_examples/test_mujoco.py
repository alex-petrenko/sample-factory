import unittest

import pytest

from sample_factory.algo.utils.misc import ExperimentStatus, EPS
from sample_factory.cfg.arguments import parse_args
from sample_factory.envs.mujoco.mujoco_utils import mujoco_available
from sample_factory.train import run_rl
from sample_factory.utils.utils import log
from sample_factory_examples.mujoco_examples.train_mujoco import register_mujoco_components


@pytest.mark.skipif(not mujoco_available(), reason='mujoco not installed')
class TestMujoco:
    """
    This test does not work if other tests used PyTorch autograd before it.
    Caused by PyTorch issue that is not easy to work around: https://github.com/pytorch/pytorch/issues/33248  # TODO: we might have fixed that by switching to multiprocessing spawn context. Need to check
    Run this test separately from other tests.

    """

    def _run_test_env(
            self, env: str = 'mujoco_ant', num_workers: int = 8, train_steps: int = 100,
            expected_reward_at_least: float = -EPS, batched_sampling: bool = False,
            serial_mode: bool = False, async_rl: bool = True, batch_size: int = 64, 
    ):
        log.debug(f'Testing with parameters {locals()}...')

        experiment_name = 'test_' + env

        register_mujoco_components()

        # test training for a few thousand frames
        cfg = parse_args(argv=['--algo=APPO', f'--env={env}', f'--experiment={experiment_name}'])
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        cfg.batched_sampling = batched_sampling
        cfg.num_workers = num_workers
        cfg.num_envs_per_worker = 2
        cfg.train_for_env_steps = train_steps
        cfg.batch_size = batch_size
        cfg.seed = 0
        cfg.device = 'cpu'

        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS

    def test_pass_env(self):
        """
        Runs tests on all envs currently passing
        """
        env_list = ['mujoco_ant', 'mujoco_halfcheetah', 'mujoco_humanoid']
        for env in env_list:
            self._run_test_env(
                env=env, num_workers=1, train_steps=100,
            )
    
    @unittest.skip('broken tests not fixed yet')
    def test_fail_action_space(self):
        """
        Currently failing tests due to incorrect action dimensions
        """
        env_list = ['mujoco_pendulum', 'mujoco_doublependulum']
        for env in env_list:
            self._run_test_env(
                env=env, num_workers=1, train_steps=50,
            )

    @unittest.skip('broken tests not fixed yet')
    def test_fail_gae(self):
        """
        Currently failing tests due to gae. Test only fails when batch size is small
        """
        env_list = ['mujoco_hopper', 'mujoco_reacher', 'mujoco_walker', 'mujoco_swimmer']
        for env in env_list:
            self._run_test_env(
                env=env, num_workers=1, train_steps=100, batch_size=64 # Setting batch size = 1024 makes it pass
            )
