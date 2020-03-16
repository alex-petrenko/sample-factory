from unittest import TestCase

from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import log


class TestQuads(TestCase):
    def test_quad_env(self):
        self.assertIsNotNone(create_env('quadrotor_single'))

        env = create_env('quadrotor_single')
        obs = env.reset()

        n_frames = 10000

        timing = Timing()
        with timing.timeit('step'):
            for i in range(n_frames):
                obs, r, d, info = env.step(env.action_space.sample())
                if d:
                    env.reset()

        log.debug('Time %s, FPS %.1f', timing, n_frames / timing.step)
