"""
Base classes for RL agent implementations with some boilerplate.

"""
import glob
import math
import os
from collections import deque
from os.path import join

import torch
from tensorboardX import SummaryWriter

from utils.decay import LinearDecay
from utils.params import Params
from utils.utils import log, ensure_dir_exists, summaries_dir, memory_consumption_mb


class TrainStatus:
    SUCCESS, FAILURE = range(2)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Agent:
    def __init__(self, params):
        self.params = params

    def initialize(self):
        pass

    def finalize(self):
        pass

    def analyze_observation(self, observation):
        """Default implementation, may be or may not be overridden."""
        return None

    def best_action(self, observation, **kwargs):
        """Must be overridden in derived classes."""
        raise NotImplementedError('Subclasses should implement {}'.format(self.best_action.__name__))


class AgentRandom(Agent):
    class Params(Params):
        def __init__(self, experiment_name):
            super(AgentRandom.Params, self).__init__(experiment_name)

        @staticmethod
        def filename_prefix():
            return 'random_'

    def __init__(self, make_env_func, params, close_env=True):
        super(AgentRandom, self).__init__(params)
        env = make_env_func()
        self.action_space = env.action_space
        if close_env:
            env.close()

    def best_action(self, *args, **kwargs):
        return self.action_space.sample()


# noinspection PyAbstractClass
class AgentLearner(Agent):
    class AgentParams(Params):
        def __init__(self, experiment_name):
            super(AgentLearner.AgentParams, self).__init__(experiment_name)
            self.initial_save_rate = 500
            self.keep_checkpoints = 5

            self.stats_episodes = 100  # how many rewards to average to measure performance

            self.gif_save_rate = 200  # number of seconds to wait before saving another gif to tensorboard
            self.gif_summary_num_envs = 2
            self.num_position_histograms = 200  # number of position heatmaps to aggregate
            self.heatmap_save_rate = 120

            self.episode_horizon = -1  # standard environment horizon

            # training process
            self.learning_rate = 1e-4
            self.train_for_steps = self.train_for_env_steps = 10 * 1000 * 1000 * 1000

    def __init__(self, params):
        super().__init__(params)

        self.device = torch.device('cuda')

        # if self.params.seed >= 0:
        #     tf.random.set_random_seed(self.params.seed)

        self.train_step = self.env_steps = 0

        self.best_avg_reward = math.nan

        self.summary_rate_decay = LinearDecay([(0, 100), (1000000, 2000), (10000000, 10000)], staircase=100)
        self.save_rate_decay = LinearDecay([(0, self.params.initial_save_rate), (1000000, 5000)], staircase=100)

        # self.initial_best_avg_reward = tf.constant(-1e3)
        # self.best_avg_reward = tf.Variable(self.initial_best_avg_reward)
        # self.total_env_steps = tf.Variable(0, dtype=tf.int64)

        # def update_best_value(best_value, new_value):
        #     return tf.assign(best_value, tf.maximum(new_value, best_value))
        # self.avg_reward_placeholder = tf.placeholder(tf.float32, [], 'new_avg_reward')
        # self.update_best_reward = update_best_value(self.best_avg_reward, self.avg_reward_placeholder)
        # self.total_env_steps_placeholder = tf.placeholder(tf.int64, [], 'new_env_steps')
        # self.update_env_steps = tf.assign(self.total_env_steps, self.total_env_steps_placeholder)

        self.position_histograms = deque([], maxlen=self.params.num_position_histograms)

        self._last_trajectory_summary = 0  # timestamp of the latest trajectory summary written
        self._last_coverage_summary = 0  # timestamp of the latest coverage summary written

        self.map_img = self.coord_limits = None

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.writer = SummaryWriter(summary_dir, flush_secs=10)

    def initialize(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found in %s. Starting from scratch!', self.params.experiment_dir())
            return

        latest_checkpoint = checkpoints[-1]
        log.warning('Loading state from checkpoint %s...', latest_checkpoint)

        if str(self.device) == 'cuda':  # the checkpoint will try to load onto the GPU storage unless specified
            checkpoint_dict = torch.load(latest_checkpoint)
        else:
            checkpoint_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)

        self._load_state(checkpoint_dict)

    def _end_of_training(self):
        return self.train_step >= self.params.train_for_steps or self.env_steps > self.params.train_for_env_steps

    def _load_state(self, checkpoint_dict):
        self.train_step = checkpoint_dict['train_step']
        self.env_steps = checkpoint_dict['env_steps']
        self.best_avg_reward = checkpoint_dict['best_avg_reward']
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def finalize(self):
        self.writer.close()

    def process_infos(self, infos):
        for i, info in enumerate(infos):
            if 'previous_histogram' in info:
                self.position_histograms.append(info['previous_histogram'])

    def _maybe_save(self):
        self.params.ensure_serialized()
        save_every = self.save_rate_decay.at(self.train_step)
        if (self.train_step + 1) % save_every == 0:
            self._save()

    def _checkpoint_dir(self):
        checkpoint_dir = join(self.params.experiment_dir(), 'checkpoint')
        return ensure_dir_exists(checkpoint_dir)

    def _get_checkpoints(self):
        checkpoints = glob.glob(join(self._checkpoint_dir(), 'checkpoint_*'))
        return sorted(checkpoints)

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'best_avg_reward': self.best_avg_reward,
        }
        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        filepath = join(self._checkpoint_dir(), f'checkpoint_{self.env_steps:012d}.pth.tar')
        log.info('Saving %s...', filepath)
        torch.save(checkpoint, filepath)

        while len(self._get_checkpoints()) > self.params.keep_checkpoints:
            oldest_checkpoint = self._get_checkpoints()[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

    def _should_write_summaries(self, step):
        summaries_every = self.summary_rate_decay.at(step)
        return (step + 1) % summaries_every == 0

    def _maybe_print(self, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.2fM ======>', self.train_step, self.env_steps / 1e6)
        log.info('Avg FPS: %.1f', fps)
        log.info('Timing: %s', t)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            return

        log.info('Avg. %d episode length: %.3f', self.params.stats_episodes, avg_length)
        best_reward_str = '' if math.isnan(self.best_avg_reward) else f'(best: {self.best_avg_reward:.3f})'
        log.info('Avg. %d episode reward: %.3f %s', self.params.stats_episodes, avg_rewards, best_reward_str)

    def _maybe_update_avg_reward(self, avg_reward, stats_num_episodes):
        if stats_num_episodes > self.params.stats_episodes:
            if math.isnan(avg_reward):
                return

            if math.isnan(self.best_avg_reward) or avg_reward > self.best_avg_reward + 1e-6:
                log.warn('New best reward %.6f (was %.6f)!', avg_reward, self.best_avg_reward)
                self.best_avg_reward = avg_reward

    def _report_train_summaries(self, stats):
        for key, scalar in stats.items():
            self.writer.add_scalar(f'train/{key}', scalar, self.env_steps)

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

    def _maybe_trajectory_summaries(self, trajectory_buffer, env_steps):
        # time_since_last = time.time() - self._last_trajectory_summary
        # if time_since_last < self.params.gif_save_rate or not trajectory_buffer.complete_trajectories:
        #     return
        #
        # start_gif_summaries = time.time()
        #
        # self._last_trajectory_summary = time.time()
        # num_envs = self.params.gif_summary_num_envs
        #
        # trajectories = [
        #     numpy_all_the_way(t.obs)[:, :, :, -3:] for t in trajectory_buffer.complete_trajectories[:num_envs]
        # ]
        # self._write_gif_summaries(tag='obs_trajectories', gif_images=trajectories, step=env_steps)
        # log.info('Took %.3f seconds to write gif summaries', time.time() - start_gif_summaries)
        pass
        # TODO!!!

    # def _write_gif_summaries(self, tag, gif_images, step, fps=12):
    #     """Logs list of input image vectors (nx[time x w h x c]) into GIFs."""
    #     def gen_gif_summary(img_stack_):
    #         img_list = np.split(img_stack_, img_stack_.shape[0], axis=0)
    #         enc_gif = encode_gif([i[0] for i in img_list], fps=fps)
    #         thwc = img_stack_.shape
    #         im_summ = tf.Summary.Image()
    #         im_summ.height = thwc[1]
    #         im_summ.width = thwc[2]
    #         im_summ.colorspace = 1  # greyscale (RGB=3, RGBA=4)
    #         im_summ.encoded_image_string = enc_gif
    #         return im_summ
    #
    #     gif_summaries = []
    #     for nr, img_stack in enumerate(gif_images):
    #         gif_summ = gen_gif_summary(img_stack)
    #         gif_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr), image=gif_summ))
    #
    #     summary = tf.Summary(value=gif_summaries)
    #     self.summary_writer.add_summary(summary, step)

    # def _maybe_coverage_summaries(self, env_steps):
    #     time_since_last = time.time() - self._last_coverage_summary
    #     if time_since_last < self.params.heatmap_save_rate:
    #         return
    #     if len(self.position_histograms) == 0:
    #         return
    #
    #     self._last_coverage_summary = time.time()
    #     self._write_position_heatmap_summaries(
    #         tag='position_coverage', step=env_steps, histograms=self.position_histograms,
    #     )
    #
    # def _write_position_heatmap_summaries(self, tag, step, histograms):
    #     summed_histogram = np.zeros_like(histograms[0])
    #     for hist in histograms:
    #         summed_histogram += hist
    #     summed_histogram += 1  # min shouldn't be 0 (for log scale)
    #
    #     fig = plt.figure(num=HEATMAP_FIGURE_ID, figsize=(4, 4))
    #     fig.clear()
    #     plt.imshow(
    #         summed_histogram.T,
    #         norm=colors.LogNorm(vmin=summed_histogram.min(), vmax=summed_histogram.max()),
    #         cmap='RdBu_r',
    #     )
    #     plt.gca().invert_yaxis()
    #     plt.colorbar()
    #
    #     summary = visualize_matplotlib_figure_tensorboard(fig, tag)
    #     self.summary_writer.add_summary(summary, step)
    #     self.summary_writer.flush()
