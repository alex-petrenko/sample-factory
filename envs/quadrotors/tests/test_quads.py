from unittest import TestCase

from envs.create_env import create_env


class TestQuads(TestCase):
    def test_quad_env(self):
        self.assertIsNotNone(create_env('quadrotor_single'))
