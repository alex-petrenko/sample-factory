import os
import shutil
from unittest import TestCase

from algorithms.utils.agent import TrainStatus
from algorithms.utils.arguments import parse_args
from enjoy_pytorch import enjoy
from train_pytorch import train
from utils.utils import experiment_dir


class TestPPO(TestCase):
    def ppo_run(self, recurrence):
        test_name = self.__class__.__name__

        argv = ['--env=doom_basic', f'--experiment={test_name}', '--algo=PPO']
        cfg = parse_args(argv)
        cfg.experiments_root = test_name
        cfg.num_envs = 16
        cfg.train_for_steps = 60
        cfg.initial_save_rate = 20
        cfg.batch_size = 32
        cfg.ppo_epochs = 2
        cfg.recurrence = recurrence
        status = train(cfg)

        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = experiment_dir(cfg=cfg)
        self.assertTrue(os.path.isdir(root_dir))

        cfg = parse_args(argv, evaluation=True)
        cfg.fps = 1e9
        enjoy(cfg, max_num_episodes=1, max_num_frames=100)
        shutil.rmtree(experiment_dir(cfg=cfg))

        self.assertFalse(os.path.isdir(root_dir))

    def test_ppo(self):
        self.ppo_run(recurrence=1)

    def test_ppo_rnn(self):
        self.ppo_run(recurrence=16)
