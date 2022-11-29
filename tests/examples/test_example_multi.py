from typing import Callable, Tuple

import pytest

from sample_factory.utils.typing import Config
from sf_examples.train_custom_multi_env import parse_custom_args, register_custom_components
from tests.examples.test_example import default_test_cfg, run_test_env


def default_multi_cfg(
    env_name: str = "my_custom_multi_env_v1", parse_args_func: Callable = parse_custom_args
) -> Tuple[Config, Config]:
    cfg, eval_cfg = default_test_cfg(env_name, parse_args_func)
    cfg.num_policies = 2
    return cfg, eval_cfg


def run_test_env_multi(cfg: Config, eval_cfg: Config, **kwargs):
    return run_test_env(
        cfg,
        eval_cfg,
        register_custom_components_func=register_custom_components,
        env_name="my_custom_multi_env_v1",
        **kwargs,
    )


class TestExampleMulti:
    @pytest.mark.parametrize("async_rl", [False, True])
    @pytest.mark.parametrize("train_steps", [1024])
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_sanity(self, async_rl: bool, train_steps: int, batched_sampling: bool):
        cfg, eval_cfg = default_multi_cfg()
        cfg.async_rl = async_rl
        cfg.train_for_env_steps = train_steps
        cfg.num_workers = 2
        cfg.batch_size = 256
        cfg.serial_mode = True
        cfg.batched_sampling = batched_sampling

        run_test_env_multi(
            cfg,
            eval_cfg,
            expected_reward_at_least=-6,  # random policy does ~-5.5, here we don't learn long enough to improve
        )

    def test_example_multi(self):
        cfg, eval_cfg = default_multi_cfg()
        cfg.async_rl = True
        cfg.serial_mode = False
        cfg.train_for_env_steps = 100000
        cfg.num_workers = 8
        cfg.batch_size = 512
        cfg.batched_sampling = False

        run_test_env_multi(
            cfg,
            eval_cfg,
            expected_reward_at_least=-0.1,  # 0 is the best we can do
        )
