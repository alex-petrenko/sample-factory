import pytest

from sf_examples.train_custom_multi_env import parse_custom_args, register_custom_components
from tests.examples.test_example import run_test_env


def run_test_env_multi(train_steps: int, num_workers: int, expected_reward_at_least: float, **kwargs):
    return run_test_env(
        num_workers=num_workers,
        experiment_name="test_example_multi",
        register_custom_components_func=register_custom_components,
        parse_args_func=parse_custom_args,
        env_name="my_custom_multi_env_v1",
        expected_reward_at_least=expected_reward_at_least,
        train_steps=train_steps,
        num_policies=2,
        **kwargs,
    )


class TestExampleMulti:
    @pytest.mark.parametrize("async_rl", [False, True])
    @pytest.mark.parametrize("train_steps", [512])
    def test_sanity(self, async_rl: bool, train_steps: int):
        run_test_env_multi(
            train_steps=train_steps,
            num_workers=1,
            batch_size=128,
            serial_mode=True,
            async_rl=async_rl,
            expected_reward_at_least=-6,  # random policy does ~-5.5, here we don't learn long enough to improve
        )

    def test_example_multi(self):
        run_test_env_multi(
            train_steps=250000,
            num_workers=8,
            batch_size=512,
            expected_reward_at_least=-0.2,  # 0 is the best we can do
            serial_mode=False,
            async_rl=True,
        )
