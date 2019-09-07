"""
Base classes for RL agent implementations with some boilerplate.

"""

import glob
import json
import math
import os
import time
from os.path import join

import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils.decay import LinearDecay
from utils.utils import log, ensure_dir_exists, summaries_dir, memory_consumption_mb, experiment_dir, cfg_file


class TrainStatus:
    SUCCESS, FAILURE = range(2)


# noinspection PyAbstractClass
class Agent:
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=42, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int, help='Save model every N steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=4, type=int, help='Number of model checkpoints to keep')

        p.add_argument('--stats_episodes', default=100, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_steps', default=int(1e10), type=int, help='Stop training after this many SGD steps')
        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop training after this many environment steps')
        p.add_argument('--train_for_seconds', default=int(1e10), type=int, help='Stop training after this many seconds')

        # observation preprocessing
        p.add_argument('--obs_subtract_mean', default=0.0, type=float, help='Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)')
        p.add_argument('--obs_scale', default=1.0, type=float, help='Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)')

        # RL
        p.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        p.add_argument(
            '--reward_scale', default=1.0, type=float,
            help=('Multiply all rewards but this factor before feeding into RL algorithm.'
                  'Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task.'
                  'Loss values become too high which requires a smaller learning rate, etc.'),
        )
        p.add_argument('--reward_clip', default=10.0, type=float, help='Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs')

        # policy size and configuration
        p.add_argument('--encoder', default='convnet_simple', type=str, help='Type of the policy head (e.g. convolutional encoder)')
        p.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)')

    def __init__(self, cfg):
        self.cfg = cfg

        if self.cfg.seed is not None:
            log.info('Settings fixed seed %d', self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.device = torch.device('cuda')

        self.train_step = self.env_steps = 0

        self.total_train_seconds = 0
        self.last_training_step = time.time()

        self.best_avg_reward = math.nan

        self.summary_rate_decay = LinearDecay([(0, 100), (1000000, 2000), (10000000, 10000)])
        self.last_summary_written = -1e9
        self.save_rate_decay = LinearDecay([(0, self.cfg.initial_save_rate), (1000000, 5000)], staircase=100)

        summary_dir = summaries_dir(experiment_dir(cfg=self.cfg))
        self.writer = SummaryWriter(summary_dir, flush_secs=10)

    def initialize(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found in %s. Starting from scratch!', experiment_dir(cfg=self.cfg))
        else:
            latest_checkpoint = checkpoints[-1]
            log.warning('Loading state from checkpoint %s...', latest_checkpoint)

            if str(self.device) == 'cuda':  # the checkpoint will try to load onto the GPU storage unless specified
                checkpoint_dict = torch.load(latest_checkpoint)
            else:
                checkpoint_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)

            self._load_state(checkpoint_dict)

        log.debug('Experiment parameters:')
        for key, value in self._cfg_dict().items():
            log.debug('\t %s: %r', key, value)

    def finalize(self):
        self.writer.close()

    def _should_end_training(self):
        end = self.train_step >= self.cfg.train_for_steps
        end |= self.env_steps > self.cfg.train_for_env_steps
        end |= self.total_train_seconds > self.cfg.train_for_seconds
        return end

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        self._maybe_save()
        self.total_train_seconds += time.time() - self.last_training_step
        self.last_training_step = time.time()

    def _on_finished_training(self):
        """This is called after normal termination, e.g. number of training steps reached."""
        log.info(
            'Finished training at train_steps %d, env_steps %d, seconds %d',
            self.train_step, self.env_steps, self.total_train_seconds,
        )
        self._save()

    def _load_state(self, checkpoint_dict):
        self.train_step = checkpoint_dict['train_step']
        self.env_steps = checkpoint_dict['env_steps']
        self.best_avg_reward = checkpoint_dict['best_avg_reward']
        self.total_train_seconds = checkpoint_dict['total_train_seconds']
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def _maybe_save(self):
        save_every = self.save_rate_decay.at(self.train_step)
        if (self.train_step + 1) % save_every == 0 or self.train_step <= 1:
            self._save()

    def _checkpoint_dir(self):
        checkpoint_dir = join(experiment_dir(cfg=self.cfg), 'checkpoint')
        return ensure_dir_exists(checkpoint_dir)

    def _get_checkpoints(self):
        checkpoints = glob.glob(join(self._checkpoint_dir(), 'checkpoint_*'))
        return sorted(checkpoints)

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'best_avg_reward': self.best_avg_reward,
            'total_train_seconds': self.total_train_seconds,
        }
        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        filepath = join(self._checkpoint_dir(), f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth')
        log.info('Saving %s...', filepath)
        torch.save(checkpoint, filepath)

        while len(self._get_checkpoints()) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self._get_checkpoints()[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

        self._save_cfg()

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def _should_write_summaries(self):
        summaries_every = self.summary_rate_decay.at(self.train_step)
        return self.train_step - self.last_summary_written > summaries_every

    def _maybe_print(self, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.2fM ======>', self.train_step, self.env_steps / 1e6)
        log.info('Avg FPS: %.1f', fps)
        log.info('Timing: %s', t)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            return

        log.info('Avg. %d episode length: %.3f', self.cfg.stats_episodes, avg_length)
        best_reward_str = '' if math.isnan(self.best_avg_reward) else f'(best: {self.best_avg_reward:.3f})'
        log.info('Avg. %d episode reward: %.3f %s', self.cfg.stats_episodes, avg_rewards, best_reward_str)

    def _maybe_update_avg_reward(self, avg_reward, stats_num_episodes):
        if stats_num_episodes > self.cfg.stats_episodes:
            if math.isnan(avg_reward):
                return

            if math.isnan(self.best_avg_reward) or avg_reward > self.best_avg_reward + 1e-6:
                log.warn('New best reward %.6f (was %.6f)!', avg_reward, self.best_avg_reward)
                self.best_avg_reward = avg_reward

    def _report_train_summaries(self, stats):
        for key, scalar in stats.items():
            self.writer.add_scalar(f'train/{key}', scalar, self.env_steps)
        self.last_summary_written = self.train_step

    def _report_basic_summaries(self, fps, avg_reward, avg_length):
        self.writer.add_scalar('0_aux/fps', fps, self.env_steps)

        memory_mb = memory_consumption_mb()
        self.writer.add_scalar('0_aux/master_process_memory_mb', float(memory_mb), self.env_steps)

        if math.isnan(avg_reward) or math.isnan(avg_length):
            # not enough data to report yet
            return

        self.writer.add_scalar('0_aux/avg_reward', float(avg_reward), self.env_steps)
        self.writer.add_scalar('0_aux/avg_length', float(avg_length), self.env_steps)

        self.writer.add_scalar('0_aux/best_reward_ever', float(self.best_avg_reward), self.env_steps)
