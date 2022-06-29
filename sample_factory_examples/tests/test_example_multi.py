import os
import shutil
import unittest
from os.path import isdir
import pytest

from sample_factory.enjoy import enjoy

from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl
from sample_factory.utils.utils import experiment_dir
from sample_factory_examples.train_custom_multi_env import register_custom_components


class TestExampleMulti:
    """
    This test does not work if other tests used PyTorch autograd before it.
    Caused by PyTorch issue that is not easy to work around: https://github.com/pytorch/pytorch/issues/33248
    Run this test separately from other tests.

    """

    @pytest.mark.skip('broken tests not fixed yet')
    def test_example_multi(self):
        experiment_name = 'test_example_multi'

        register_custom_components()

        # test training for a few thousand frames
        cfg = parse_args(argv=['--algo=APPO', '--env=my_custom_multi_env_v1', f'--experiment={experiment_name}'])
        cfg.num_workers = 6
        cfg.train_for_env_steps = 350000
        cfg.save_every_sec = 1
        cfg.decorrelate_experience_max_seconds = 0
        cfg.seed = 0
        cfg.device = 'cpu'
        cfg.num_policies = 2

        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS

        # then test the evaluation of the saved model
        cfg = parse_args(
            argv=['--algo=APPO', '--env=my_custom_multi_env_v1', f'--experiment={experiment_name}', '--device=cpu'],
            evaluation=True,
        )
        status, avg_reward = enjoy(cfg, max_num_frames=200)

        directory = experiment_dir(cfg=cfg)
        assert isdir(directory)
        shutil.rmtree(directory, ignore_errors=True)

        assert status == ExperimentStatus.SUCCESS

        # not sure if we should check it here, it's optional
        # maybe a longer test where it actually has a chance to converge
        assert avg_reward > -1.3
