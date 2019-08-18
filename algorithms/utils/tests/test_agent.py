import os
import shutil
from unittest import TestCase

from algorithms.utils.agent import Agent
from algorithms.utils.arguments import parse_args
from utils.utils import experiment_dir


class TestAgent(TestCase):
    def test_checkpoints(self):
        cfg = parse_args(argv=['--experiment=__test__', '--env=e', '--algo=a'])

        agent = Agent(cfg)
        agent.initialize()

        for _ in range(10):
            agent.train_step += 1
            agent.env_steps += 10
            agent._save()

        curr_train_step = agent.train_step
        curr_env_step = agent.env_steps

        agent.initialize()
        self.assertEqual(agent.train_step, curr_train_step)
        self.assertEqual(agent.env_steps, curr_env_step)

        agent._save()

        agent_dir = experiment_dir(cfg=cfg)
        shutil.rmtree(agent_dir)
        self.assertFalse(os.path.isdir(agent_dir))
