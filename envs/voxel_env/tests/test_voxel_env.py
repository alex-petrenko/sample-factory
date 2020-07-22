import unittest
from unittest import TestCase

from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env
from envs.env_utils import voxel_env_available
from utils.utils import log


class TestVoxelEnv(TestCase):
    @unittest.skipUnless(voxel_env_available(), 'VoxelEnv not installed')
    def test_voxel_env(self):
        env_name = 'voxel_env_v1'
        env = create_env(env_name, cfg=default_cfg(env=env_name))
        log.info('Env action space: %r', env.action_space)
        log.info('Env obs space: %r', env.observation_space)

        env.reset()
        total_rew = 0
        for i in range(1000):
            obs, rew, done, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])
            total_rew += sum(rew)

        log.info('Total rew: %.3f', total_rew)
