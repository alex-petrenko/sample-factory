import pytest

from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import parse_args
from sample_factory_examples.mujoco_examples.mujoco_utils import mujoco_available
from sample_factory.train import run_rl
from sample_factory.utils.utils import experiment_dir, log

import shutil
from os.path import isdir


@pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed")
class TestMujoco:
    @pytest.fixture(scope="class", autouse=True)
    def register_mujoco_fixture(self):
        from sample_factory_examples.mujoco_examples.train_mujoco import register_mujoco_components

        register_mujoco_components()

    @staticmethod
    def _run_test_env(
        env: str = "mujoco_ant",
        num_workers: int = 8,
        train_steps: int = 128,
        batched_sampling: bool = False,
        serial_mode: bool = True,
        async_rl: bool = False,
        batch_size: int = 64,
        rollout: int = 8,
    ):
        log.debug(f"Testing with parameters {locals()}...")
        assert train_steps > batch_size, "We need sufficient number of steps to accumulate at least one batch"

        experiment_name = "test_" + env

        cfg = parse_args(argv=["--algo=APPO", f"--env={env}", f"--experiment={experiment_name}"])
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        cfg.batched_sampling = batched_sampling
        cfg.num_workers = num_workers
        cfg.num_envs_per_worker = 2
        cfg.train_for_env_steps = train_steps
        cfg.batch_size = batch_size
        cfg.rollout = rollout
        cfg.decorrelate_envs_on_one_worker = False
        cfg.decorrelate_experience_max_seconds = 0
        cfg.seed = 0
        cfg.device = "cpu"

        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS

        directory = experiment_dir(cfg=cfg)
        assert isdir(directory)
        shutil.rmtree(directory, ignore_errors=True)


    @pytest.mark.parametrize("env_name", ["mujoco_ant", "mujoco_halfcheetah", "mujoco_humanoid", "mujoco_hopper", "mujoco_reacher", "mujoco_walker", "mujoco_swimmer"])
    @pytest.mark.parametrize("num_workers", [1, 8])
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_basic_envs(self, env_name, batched_sampling, num_workers):
        self._run_test_env(env=env_name, num_workers=num_workers, batched_sampling=batched_sampling)

    @pytest.mark.parametrize("env_name", ["mujoco_pendulum", "mujoco_doublependulum"])
    @pytest.mark.parametrize("num_workers", [1, 8])
    def test_single_action_envs_batched(self, env_name, num_workers):
        """These envs only have a single action and might cause unique problems with 0-D vs 1-D tensors."""
        self._run_test_env(env=env_name, num_workers=num_workers, batched_sampling=True)

    @pytest.mark.parametrize("env_name", ["mujoco_pendulum", "mujoco_doublependulum"])
    @pytest.mark.parametrize("num_workers", [1, 8])
    def test_single_action_envs_non_batched(self, env_name, num_workers):
        """These envs only have a single action and might cause unique problems with 0-D vs 1-D tensors."""
        self._run_test_env(env=env_name, num_workers=num_workers, batched_sampling=False)
