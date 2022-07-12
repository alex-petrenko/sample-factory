import shutil
from os.path import isdir

import pytest

from sample_factory.algo.utils.misc import EPS, ExperimentStatus
from sample_factory.enjoy import enjoy
from sample_factory.train import run_rl
from sample_factory.utils.utils import experiment_dir, log
from sf_examples.train_custom_env_custom_model import parse_custom_args, register_custom_components


class TestExample:
    """
    This test does not work if other tests used PyTorch autograd before it.
    Caused by PyTorch issue that is not easy to work around: https://github.com/pytorch/pytorch/issues/33248  # TODO: we might have fixed that by switching to multiprocessing spawn context. Need to check
    Run this test separately from other tests.

    """

    @staticmethod
    def _run_test_env(
        num_actions: int = 10,
        num_workers: int = 8,
        train_steps: int = 128,
        batch_size: int = 64,
        expected_reward_at_least: float = -EPS,
        expected_reward_at_most: float = 100,
        batched_sampling: bool = False,
        serial_mode: bool = False,
        async_rl: bool = True,
    ):
        log.debug(f"Testing with parameters {locals()}...")

        experiment_name = "test_example"

        register_custom_components()

        # test training for a few thousand frames
        cfg = parse_custom_args(argv=["--algo=APPO", "--env=my_custom_env_v1", f"--experiment={experiment_name}"])
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        cfg.batched_sampling = batched_sampling
        cfg.custom_env_num_actions = num_actions
        cfg.num_workers = num_workers
        cfg.num_envs_per_worker = 2
        cfg.batch_size = batch_size
        cfg.train_for_env_steps = train_steps
        cfg.save_every_sec = 1
        cfg.decorrelate_experience_max_seconds = 0
        cfg.decorrelate_envs_on_one_worker = False
        cfg.seed = 0
        cfg.device = "cpu"
        cfg.learning_rate = 1e-3
        cfg.normalize_input = True
        cfg.normalize_returns = True
        cfg.with_vtrace = False

        directory = experiment_dir(cfg=cfg, mkdir=False)
        if isdir(directory):
            # remove any previous unfinished test dirs so they don't interfere with this test
            shutil.rmtree(directory, ignore_errors=True)

        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS

        # then test the evaluation of the saved model
        cfg = parse_custom_args(
            argv=["--algo=APPO", "--env=my_custom_env_v1", f"--experiment={experiment_name}"],
            evaluation=True,
        )
        cfg.device = "cpu"
        status, avg_reward = enjoy(cfg, max_num_frames=1000)

        assert isdir(directory)
        shutil.rmtree(directory, ignore_errors=True)

        assert status == ExperimentStatus.SUCCESS
        assert avg_reward >= expected_reward_at_least
        assert avg_reward <= expected_reward_at_most

    @pytest.mark.parametrize("num_actions", [1, 10])
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_sanity_1(self, num_actions, batched_sampling):
        """
        Run the test env in various configurations just to make sure nothing crashes or throws exceptions.
        """
        self._run_test_env(
            num_actions=num_actions,
            num_workers=1,
            train_steps=50,
            batched_sampling=batched_sampling,
        )

    @pytest.mark.parametrize("serial_mode", [False, True])
    @pytest.mark.parametrize("async_rl", [False, True])
    def test_sanity_2(self, serial_mode, async_rl):
        self._run_test_env(
            num_actions=10,
            num_workers=1,
            train_steps=50,
            batched_sampling=False,
            serial_mode=serial_mode,
            async_rl=async_rl,
        )

    def test_full_run(self):
        """Actually train this little env and expect some reward."""
        self._run_test_env(
            num_actions=10,
            num_workers=8,
            train_steps=100000,
            batch_size=256,
            expected_reward_at_least=80,
            expected_reward_at_most=100,
            batched_sampling=False,
            serial_mode=False,
            async_rl=True,
        )
