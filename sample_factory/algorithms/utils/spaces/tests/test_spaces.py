from unittest import TestCase

from sample_factory.algorithms.utils.spaces.discretized import Discretized


class TestSpaces(TestCase):
    def test_discretized(self):
        n = 11
        min_action = -10.0
        max_action = 10.0
        space = Discretized(n, min_action, max_action)
        random_action = space.sample()
        self.assertGreaterEqual(random_action, 0)
        self.assertLess(random_action, n)

        expected_value = min_action
        step = (max_action - min_action) / (n - 1)
        for action in range(n):
            continuous_action = space.to_continuous(action)
            self.assertAlmostEqual(continuous_action, expected_value)
            expected_value += step
