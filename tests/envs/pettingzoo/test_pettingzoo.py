import shutil
from os.path import isdir

import pytest

from sample_factory.algo.utils.context import reset_global_context
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.train import run_rl
from sample_factory.utils.utils import log
from sf_examples.train_pettingzoo_env import make_pettingzoo_env, parse_custom_args, register_custom_components
from tests.envs.utils import eval_env_performance
from tests.export_onnx_utils import check_export_onnx
from tests.utils import clean_test_dir


class TestPettingZooEnv:
    @pytest.fixture(scope="class", autouse=True)
    def register_pettingzoo_fixture(self):
        register_custom_components()
        yield  # this is where the actual test happens
        reset_global_context()

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        return make_pettingzoo_env("tictactoe_v3", cfg=parse_custom_args(argv=["--algo=APPO", "--env=tictactoe_v3"]))

    def test_pettingzoo_performance(self):
        eval_env_performance(self.make_env, "pettingzoo")

    @staticmethod
    def _run_test_env(
        env: str = "tictactoe_v3",
        num_workers: int = 2,
        train_steps: int = 512,
        batched_sampling: bool = False,
        serial_mode: bool = True,
        async_rl: bool = False,
        batch_size: int = 256,
    ):
        log.debug(f"Testing with parameters {locals()}...")
        assert train_steps > batch_size, "We need sufficient number of steps to accumulate at least one batch"

        experiment_name = "test_" + env

        def parse_args(evaluation: bool = False):
            cfg = parse_custom_args(
                argv=["--algo=APPO", f"--env={env}", f"--experiment={experiment_name}"], evaluation=evaluation
            )
            cfg.serial_mode = serial_mode
            cfg.async_rl = async_rl
            cfg.batched_sampling = batched_sampling
            cfg.num_workers = num_workers
            cfg.train_for_env_steps = train_steps
            cfg.batch_size = batch_size
            cfg.decorrelate_envs_on_one_worker = False
            cfg.seed = 0
            cfg.device = "cpu"
            cfg.eval_deterministic = True
            return cfg

        cfg = parse_args(env)
        directory = clean_test_dir(cfg)
        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS
        assert isdir(directory)

        try:
            cfg = parse_args(evaluation=True)
            check_export_onnx(cfg)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_basic_envs(self, batched_sampling):
        self._run_test_env(batched_sampling=batched_sampling)
