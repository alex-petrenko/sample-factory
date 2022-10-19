import shutil
from os.path import isdir
from typing import Callable, Tuple

import pytest

from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.sampling.sampler import SerialSampler
from sample_factory.algo.utils.context import reset_global_context
from sample_factory.algo.utils.make_env import SequentialVectorizeWrapper
from sample_factory.algo.utils.misc import EPS, ExperimentStatus
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import (
    RewardShapingInterface,
    TrainingInfoInterface,
    find_training_info_interface,
    find_wrapper_interface,
)
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir, log
from sf_examples.train_custom_env_custom_model import parse_custom_args, register_custom_components


def default_test_cfg(
    env_name: str = "my_custom_env_v1", parse_args_func: Callable = parse_custom_args
) -> Tuple[Config, Config]:
    argv = ["--algo=APPO", f"--env={env_name}", "--experiment=test_example"]
    cfg = parse_args_func(argv=argv)
    cfg.num_workers = 8
    cfg.num_envs_per_worker = 2
    cfg.train_for_env_steps = 128
    cfg.batch_size = 64
    cfg.batched_sampling = False
    cfg.async_rl = True
    cfg.save_every_sec = 4
    cfg.decorrelate_experience_max_seconds = 0
    cfg.decorrelate_envs_on_one_worker = False
    cfg.seed = 0
    cfg.device = "cpu"
    cfg.learning_rate = 1e-3
    cfg.normalize_input = True
    cfg.normalize_returns = True
    cfg.with_vtrace = False

    eval_cfg = parse_args_func(argv=argv, evaluation=True)
    eval_cfg.device = "cpu"
    eval_cfg.max_num_frames = 1000
    eval_cfg.no_render = True
    return cfg, eval_cfg


def run_test_env(
    cfg: Config,
    eval_cfg: Config,
    expected_reward_at_least: float = -EPS,
    expected_reward_at_most: float = 100,
    check_envs: bool = False,
    register_custom_components_func: Callable = register_custom_components,
    env_name: str = "my_custom_env_v1",
):
    log.debug(f"Testing with parameters {locals()}...")

    register_custom_components_func()

    directory = experiment_dir(cfg=cfg, mkdir=False)
    if isdir(directory):
        # remove any previous unfinished test dirs so they don't interfere with this test
        shutil.rmtree(directory, ignore_errors=True)

    cfg, runner = make_runner(cfg)
    if check_envs:
        runner.update_training_info_every_sec = 0.1  # so we can test training info updates
    status = runner.init()
    assert status == ExperimentStatus.SUCCESS
    status = runner.run()
    assert status == ExperimentStatus.SUCCESS

    # then test the evaluation of the saved model
    status, avg_reward = enjoy(eval_cfg)
    log.debug(f"Test reward: {avg_reward:.4f}")

    assert isdir(directory)
    shutil.rmtree(directory, ignore_errors=True)

    assert status == ExperimentStatus.SUCCESS
    assert avg_reward >= expected_reward_at_least
    assert avg_reward <= expected_reward_at_most

    if cfg.serial_mode and check_envs:
        # we can directly access the envs and check things in serial mode
        assert runner.sampler is not None
        assert isinstance(runner.sampler, SerialSampler)
        sampler: SerialSampler = runner.sampler
        env_runner = sampler.rollout_workers[0].env_runners[0]
        envs = []
        if isinstance(env_runner, BatchedVectorEnvRunner):
            assert env_runner.vec_env is not None
            assert isinstance(env_runner.vec_env, SequentialVectorizeWrapper)
            envs = env_runner.vec_env.envs
        elif isinstance(env_runner, NonBatchedVectorEnvRunner):
            envs = env_runner.envs

        for env in envs:
            env_train_info = find_training_info_interface(env)
            assert isinstance(env_train_info, TrainingInfoInterface)
            assert "approx_total_training_steps" in env_train_info.training_info

            env_rew_shaping = find_wrapper_interface(env, RewardShapingInterface)
            assert isinstance(env_rew_shaping, RewardShapingInterface)

    reset_global_context()


class TestExample:
    @pytest.mark.parametrize("num_actions", [1, 10])
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_sanity_1(self, num_actions: int, batched_sampling: bool):
        """
        Run the test env in various configurations just to make sure nothing crashes or throws exceptions.
        """
        cfg, eval_cfg = default_test_cfg()
        cfg.custom_env_num_actions = eval_cfg.custom_env_num_actions = num_actions
        cfg.num_workers = 1
        cfg.train_for_env_steps = 50
        cfg.batched_sampling = batched_sampling
        run_test_env(cfg, eval_cfg)

    @pytest.mark.parametrize("serial_mode", [False, True])
    @pytest.mark.parametrize("async_rl", [False, True])
    def test_sanity_2(self, serial_mode: bool, async_rl: bool):
        cfg, eval_cfg = default_test_cfg()
        cfg.num_workers = 1
        cfg.train_for_env_steps = 50
        cfg.batched_sampling = False
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        run_test_env(cfg, eval_cfg)

    @pytest.mark.parametrize("batched_sampling", [True, False])
    def test_chk_envs(self, batched_sampling: bool):
        cfg, eval_cfg = default_test_cfg()
        cfg.num_workers = 1
        cfg.worker_num_splits = 1
        cfg.train_for_env_steps = 250
        cfg.custom_env_episode_len = 50
        cfg.batched_sampling = batched_sampling
        cfg.serial_mode = True
        cfg.async_rl = False
        run_test_env(cfg, eval_cfg, check_envs=True)

    def test_full_run(self):
        """Actually train this little env and expect some reward."""
        cfg, eval_cfg = default_test_cfg()
        cfg.train_for_env_steps = 90000
        cfg.batch_size = 256
        cfg.batched_sampling = False
        cfg.serial_mode = False
        cfg.async_rl = True

        run_test_env(
            cfg,
            eval_cfg,
            expected_reward_at_least=80,
            expected_reward_at_most=100,
        )
