import os
import shutil
from os.path import join
from unittest import TestCase

from algorithms.ppo.agent_ppo import AgentPPO
from algorithms.ppo.enjoy_ppo import enjoy
from algorithms.ppo.train_ppo import train
from algorithms.utils.agent import TrainStatus
from algorithms.utils.arguments import parse_args
from utils.utils import experiments_dir


class TestPPO(TestCase):
    def ppo_run(self, recurrence):
        test_name = self.__class__.__name__

        argv = ['--env=doom_basic', f'--experiment={test_name}']
        args, params = parse_args(AgentPPO.Params, argv=argv)
        params.experiments_root = test_name
        params.num_envs = 16
        params.train_for_steps = 60
        params.initial_save_rate = 20
        params.batch_size = 32
        params.ppo_epochs = 2
        params.recurrence = recurrence
        status = train(args, params)

        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        eval_args, _ = parse_args(AgentPPO.Params, argv=argv, evaluation=True)
        eval_args.fps = 1e9
        enjoy(eval_args, params, max_num_episodes=1, max_num_frames=100)
        shutil.rmtree(join(experiments_dir(), params.experiments_root))

        self.assertFalse(os.path.isdir(root_dir))

    def test_ppo(self):
        self.ppo_run(recurrence=1)

    def test_ppo_rnn(self):
        self.ppo_run(recurrence=16)
