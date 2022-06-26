import os
import shutil
import unittest
from os.path import isdir
from unittest import TestCase

from sample_factory.algo.utils.misc import ExperimentStatus, EPS
from sample_factory.enjoy import enjoy
from sample_factory.train import run_rl
from sample_factory_examples.train_custom_env_custom_model import register_custom_components, custom_parse_args
from sample_factory.utils.utils import experiment_dir, log


@unittest.skipIf(
    'SKIP_TESTS_THAT_REQUIRE_A_SEPARATE_PROCESS' in os.environ,
    'this test should be executed in a separate process because of how PyTorch works: '
    'https://github.com/pytorch/pytorch/issues/33248',
)
class TestExample(TestCase):
    """
    This test does not work if other tests used PyTorch autograd before it.
    Caused by PyTorch issue that is not easy to work around: https://github.com/pytorch/pytorch/issues/33248  # TODO: we might have fixed that by switching to multiprocessing spawn context. Need to check
    Run this test separately from other tests.

    """

    def _run_test_env(
            self, num_actions: int = 10, num_workers: int = 8, train_steps: int = 100,
            expected_reward_at_least: float = -EPS, batched_sampling: bool = False,
            serial_mode: bool = False, async_rl: bool = True,
    ):
        log.debug(f'Testing with parameters {locals()}...')

        experiment_name = 'test_example'

        register_custom_components()

        # test training for a few thousand frames
        cfg = custom_parse_args(argv=['--algo=APPO', '--env=my_custom_env_v1', f'--experiment={experiment_name}'])
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        cfg.batched_sampling = batched_sampling
        cfg.custom_env_num_actions = num_actions
        cfg.num_workers = num_workers
        cfg.num_envs_per_worker = 2
        cfg.train_for_env_steps = train_steps
        cfg.save_every_sec = 1
        cfg.decorrelate_experience_max_seconds = 0
        cfg.decorrelate_envs_on_one_worker = False
        cfg.seed = 0
        cfg.device = 'cpu'

        status = run_rl(cfg)
        self.assertEqual(status, ExperimentStatus.SUCCESS)

        # then test the evaluation of the saved model
        cfg = custom_parse_args(
            argv=['--algo=APPO', '--env=my_custom_env_v1', f'--experiment={experiment_name}'],
            evaluation=True,
        )
        cfg.device = 'cpu'
        status, avg_reward = enjoy(cfg, max_num_frames=1000)

        directory = experiment_dir(cfg=cfg)
        self.assertTrue(isdir(directory))
        shutil.rmtree(directory, ignore_errors=True)
        # self.assertFalse(isdir(directory))

        self.assertEqual(status, ExperimentStatus.SUCCESS)

        # not sure if we should check it here, it's optional
        # maybe a longer test where it actually has a chance to converge
        self.assertGreaterEqual(avg_reward, expected_reward_at_least)

    def test_sanity(self):
        """
        Run the test env in various configurations just to make sure nothing crashes or throws exceptions.
        """
        for num_actions in [1, 10]:
            for batched_sampling in [False, True]:
                self._run_test_env(
                    num_actions=num_actions, num_workers=1, train_steps=50, batched_sampling=batched_sampling,
                )

        for serial_mode in [False, True]:
            for async_rl in [False, True]:
                self._run_test_env(
                    num_actions=10, num_workers=1, train_steps=50, batched_sampling=False,
                    serial_mode=serial_mode, async_rl=async_rl,
                )

    def test_full_run(self):
        """Actually train this little env and expect some reward."""
        self._run_test_env(
            num_actions=10,
            num_workers=8,
            train_steps=100000,
            expected_reward_at_least=60,
            batched_sampling=False,
            serial_mode=False,
            async_rl=True,
        )
