import gc
import os
import shutil
import time
from os.path import join
from unittest import TestCase

import numpy as np
import tensorflow as tf

from algorithms.utils.agent import TrainStatus
from algorithms.baselines.ppo.agent_ppo import AgentPPO, PPOBuffer, ActorCritic
from algorithms.baselines.ppo.enjoy_ppo import enjoy
from algorithms.baselines.ppo.ppo_utils import parse_args_ppo
from algorithms.baselines.ppo.train_ppo import train
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.utils.tf_utils import placeholder_from_space, placeholders
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.utils import log, AttrDict, experiments_dir


class TestPPO(TestCase):
    def ppo_train_run(self, env_name=None):
        test_dir_name = self.__class__.__name__

        args, params = parse_args_ppo(AgentPPO.Params)
        if env_name is not None:
            args.env = env_name
        params.experiments_root = test_dir_name
        params.num_envs = 16
        params.train_for_steps = 60
        params.initial_save_rate = 20
        params.batch_size = 32
        params.ppo_epochs = 2
        params.use_env_map = False
        status = train(params, args.env)
        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        enjoy(params, args.env, max_num_episodes=1, max_num_frames=50, fps=1000)
        shutil.rmtree(join(experiments_dir(), params.experiments_root))

        self.assertFalse(os.path.isdir(root_dir))

    def test_ppo_train_run(self):
        self.ppo_train_run()

    def test_ppo_train_run_goal(self):
        self.ppo_train_run(env_name='doom_maze_goal')

    def test_buffer_batches(self):
        obs_size, num_envs, rollout, batch_size = 16, 10, 100, 50

        buff = PPOBuffer()

        # fill buffer with fake data
        for item in buff.__dict__.keys():
            buff.__dict__[item] = np.ones((rollout, num_envs))
        buff.obs = np.ones((rollout, num_envs, obs_size))
        buff.dones = np.zeros_like(buff.dones)
        buff.values = np.append(buff.values, np.ones((1, num_envs)), axis=0)

        buff.finalize_batch(0.999, 0.99)
        buff.shuffle()

        self.assertEqual(buff.obs.shape, (rollout * num_envs, obs_size))
        self.assertEqual(buff.rewards.shape, (rollout * num_envs, ))


class TestPPOPerformance(TestCase):
    @staticmethod
    def setup_graph(env, params, use_dataset):
        tf.reset_default_graph()

        step = tf.Variable(0, trainable=False, dtype=tf.int64, name='step')

        ph_observations = placeholder_from_space(env.observation_space)
        ph_actions = placeholder_from_space(env.action_space)
        ph_old_actions_probs, ph_advantages, ph_returns = placeholders(None, None, None)

        if use_dataset:
            dataset = tf.data.Dataset.from_tensor_slices((
                ph_observations,
                ph_actions,
                ph_old_actions_probs,
                ph_advantages,
                ph_returns,
            ))
            dataset = dataset.batch(params.batch_size)
            dataset = dataset.prefetch(10)
            iterator = dataset.make_initializable_iterator()
            observations, act, old_action_probs, adv, ret = iterator.get_next()
        else:
            observations = ph_observations
            act, old_action_probs, adv, ret = ph_actions, ph_old_actions_probs, ph_advantages, ph_returns

        actor_critic = ActorCritic(env, observations, params)
        env.close()

        objectives = AgentPPO.add_ppo_objectives(actor_critic, act, old_action_probs, adv, ret, params, step)
        train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(objectives.actor_loss, global_step=step)

        return AttrDict(locals())

    def train_feed_dict(self, env, data, params, use_gpu):
        num_batches = len(data.obs) // params.batch_size
        g = self.setup_graph(env, params, use_dataset=False)

        config = tf.ConfigProto(device_count={'GPU': 100 if use_gpu else 0})
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(params.ppo_epochs):
                epoch_starts = time.time()
                for i in range(num_batches):
                    start, end = i * params.batch_size, (i + 1) * params.batch_size
                    kl, _ = sess.run(
                        [g.objectives.sample_kl, g.train_op],
                        feed_dict={
                            g.ph_observations: data.obs[start:end],
                            g.ph_actions: data.act[start:end],
                            g.ph_old_actions_probs: data.old_prob[start:end],
                            g.ph_advantages: data.adv[start:end],
                            g.ph_returns: data.ret[start:end],
                        })
                    del kl
                time_per_epoch = time.time() - epoch_starts

                log.debug(
                    'Feed dict gpu %r: took %.3f seconds per epoch (%d batches, %d samples)',
                    use_gpu, time_per_epoch, num_batches, len(data.obs),
                )

        tf.reset_default_graph()
        gc.collect()

    def train_dataset(self, env, data, params, use_gpu):
        num_batches = len(data.obs) // params.batch_size
        g = self.setup_graph(env, params, use_dataset=True)

        config = tf.ConfigProto(device_count={'GPU': 100 if use_gpu else 0})
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(params.ppo_epochs):
                epoch_starts = time.time()
                sess.run(
                    g.iterator.initializer,
                    feed_dict={
                        g.ph_observations: data.obs,
                        g.ph_actions: data.act,
                        g.ph_old_actions_probs: data.old_prob,
                        g.ph_advantages: data.adv,
                        g.ph_returns: data.ret,
                    })

                while True:
                    try:
                        kl, _ = sess.run([g.objectives.sample_kl, g.train_op])
                        del kl
                    except tf.errors.OutOfRangeError:
                        break

                time_per_epoch = time.time() - epoch_starts

                log.debug(
                    'tf.data gpu %r: took %.3f seconds per epoch (%d batches, %d samples)',
                    use_gpu, time_per_epoch, num_batches, len(data.obs),
                )

        tf.reset_default_graph()
        gc.collect()

    def test_performance(self):
        params = AgentPPO.Params('test_performance')
        params.ppo_epochs = 2
        params.rollout = 16
        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))

        observation_shape = env.observation_space.shape
        experience_size = params.num_envs * params.rollout

        # generate random data
        data = AttrDict()
        data.obs = np.random.normal(size=(experience_size,) + observation_shape)
        data.act = np.random.randint(0, 3, size=[experience_size])
        data.old_prob = np.random.uniform(0, 1, size=[experience_size])
        data.adv = np.random.normal(size=[experience_size])
        data.ret = np.random.normal(size=[experience_size])

        self.train_feed_dict(env, data, params, use_gpu=False)
        self.train_feed_dict(env, data, params, use_gpu=True)
        self.train_dataset(env, data, params, use_gpu=False)
        self.train_dataset(env, data, params, use_gpu=True)

        env.close()
