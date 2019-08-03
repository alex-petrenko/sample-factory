import os
import shutil
from unittest import TestCase

from algorithms.utils.agent import AgentLearner


class TestAgent(TestCase):
    def test_checkpoints(self):
        params = AgentLearner.AgentParams(self.__class__.__name__)
        agent = AgentLearner(params)
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

        agent_dir = params.experiment_dir()
        shutil.rmtree(agent_dir)
        self.assertFalse(os.path.isdir(agent_dir))
