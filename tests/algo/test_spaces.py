import pytest

from sample_factory.algo.utils.spaces.discretized import Discretized


class TestSpaces:
    @pytest.mark.parametrize("n", [11])
    @pytest.mark.parametrize("min_action", [-10.0])
    @pytest.mark.parametrize("max_action", [10.0])
    def test_discretized(self, n, min_action, max_action):
        space = Discretized(n, min_action, max_action)
        random_action = space.sample()
        assert random_action >= 0
        assert random_action < n

        expected_value = min_action
        step = (max_action - min_action) / (n - 1)
        for action in range(n):
            continuous_action = space.to_continuous(action)
            assert pytest.approx(continuous_action) == expected_value
            expected_value += step
