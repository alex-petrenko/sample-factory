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
    def test_sanity(self, async_rl: bool):
        run_test_env_multi(
            train_steps=512,
            num_workers=1,
            batch_size=128,
            serial_mode=False,
            async_rl=async_rl,
            expected_reward_at_least=-2,
        )

    def test_example_multi(self):
        run_test_env_multi(
            train_steps=250000,
            num_workers=8,
            batch_size=512,
            expected_reward_at_least=-0.4,  # 0 is the best we can do (would be nice to figure out why it does not converge all the way to 0)
            serial_mode=False,
            async_rl=True,
        )
