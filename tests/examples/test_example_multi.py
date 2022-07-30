import pytest

from sf_examples.train_custom_multi_env import register_custom_components
from tests.examples.test_example import run_test_env


def run_test_env_multi(train_steps: int, num_workers: int, expected_reward_at_least: float, **kwargs):
    return run_test_env(
        num_workers=num_workers,
        experiment_name="test_example_multi",
        register_custom_components_func=register_custom_components,
        env_name="my_custom_multi_env_v1",
        expected_reward_at_least=expected_reward_at_least,
        train_steps=train_steps,
        num_policies=2,
        **kwargs,
    )


class TestExampleMulti:
    def test_sanity(self):
        run_test_env_multi(
            train_steps=128,
            num_workers=1,
            batch_size=128,
            serial_mode=True,
            async_rl=False,
            expected_reward_at_least=-1000,
        )

    @pytest.mark.skip(reason="TODO: fix this test")
    def test_example_multi(self):
        run_test_env_multi(
            train_steps=350000,
            num_workers=8,
            expected_reward_at_least=-1.3,
            serial_mode=True,
            async_rl=False,
        )  # TODO disable serial mode
