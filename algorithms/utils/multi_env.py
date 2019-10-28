import threading
import time
from multiprocessing import Process, JoinableQueue
from enum import Enum
import copy
from queue import Queue, Empty

import numpy as np

from algorithms.utils.algo_utils import list_to_string
from utils.utils import log, AttrDict


def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.exception(msg)


class MsgType(Enum):
    INIT, TERMINATE, RESET, STEP_REAL, STEP_REAL_RESET, STEP_IMAGINED, INFO = range(7)


class _MultiEnvWorker:
    """Helper class for the MultiEnv."""

    def __init__(self, env_indices, make_env_func, use_multiprocessing):
        self._verbose = False

        self.make_env_func = make_env_func
        self.env_indices = env_indices
        self.timestep = 0

        if use_multiprocessing:
            self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
            self.process = Process(target=self.start, daemon=False)
        else:
            log.debug('Not using multiprocessing!')
            self.task_queue, self.result_queue = Queue(), Queue()
            # shouldn't be named "process" here, but who cares
            self.process = threading.Thread(target=self.start)

        self.is_multiagent = False
        self.process.start()

    def _init(self, envs):
        log.info('Initializing envs %s...', list_to_string(self.env_indices))
        worker_index = self.env_indices[0] // len(self.env_indices)
        for i in self.env_indices:
            env_config = AttrDict({'worker_index': worker_index, 'vector_index': i - self.env_indices[0]})
            env = self.make_env_func(env_config)
            env.seed(i)
            env.reset()
            if hasattr(env, 'num_agents') and env.num_agents > 1:
                self.is_multiagent = True
            envs.append(env)
            time.sleep(0.01)

    def _terminate(self, real_envs, imagined_envs):
        if self._verbose:
            log.info('Stop worker %s...', list_to_string(self.env_indices))
        for e in real_envs:
            e.close()
        if imagined_envs is not None:
            for imagined_env in imagined_envs:
                imagined_env.close()

        if self._verbose:
            log.info('Worker %s terminated!', list_to_string(self.env_indices))

    @staticmethod
    def _get_info(env):
        info = {}
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_info_all'):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def start(self):
        real_envs = []
        imagined_envs = None

        timing = AttrDict({'copying': 0, 'prediction': 0})

        while True:
            actions, msg_type = safe_get(self.task_queue)

            if msg_type == MsgType.INIT:
                self._init(real_envs)
                self.task_queue.task_done()
                continue

            if msg_type == MsgType.TERMINATE:
                self._terminate(real_envs, imagined_envs)
                self.task_queue.task_done()
                break

            # handling actual workload
            envs = real_envs
            if msg_type == MsgType.RESET or msg_type == MsgType.STEP_REAL or msg_type == MsgType.STEP_REAL_RESET:
                if imagined_envs is not None:
                    for imagined_env in imagined_envs:
                        imagined_env.close()
                imagined_envs = None
            elif msg_type == MsgType.INFO:
                pass
            else:

                if imagined_envs is None:
                    # initializing new prediction, let's report timing for the previous one
                    if timing.prediction > 0 and self._verbose:
                        log.debug(
                            'Multi-env copy took %.6f s, prediction took %.6f s',
                            timing.copying, timing.prediction,
                        )

                    timing.prediction = 0
                    timing.copying = time.time()

                    imagined_envs = []
                    # we expect a list of actions for every environment in this worker (list of lists)
                    assert len(actions) == len(real_envs)
                    for env_idx in range(len(actions)):
                        for _ in actions[env_idx]:
                            imagined_env = copy.deepcopy(real_envs[env_idx])
                            imagined_envs.append(imagined_env)
                    timing.copying = time.time() - timing.copying

                envs = imagined_envs
                actions = np.asarray(actions).flatten()

            if msg_type == MsgType.RESET:
                results = [env.reset() for env in envs]
            elif msg_type == MsgType.INFO:
                results = [self._get_info(env) for env in envs]
            else:
                assert len(envs) == len(actions)

                reset = [False] * len(actions)
                if msg_type == MsgType.STEP_REAL_RESET:
                    actions, reset = zip(*actions)

                # Collect obs, reward, done, and info
                prediction_start = time.time()
                results = [env.step(action) for env, action in zip(envs, actions)]
                self.timestep += 1

                # pack results per-env
                results = np.split(np.array(results), len(real_envs))

                if msg_type == MsgType.STEP_IMAGINED:
                    timing.prediction += time.time() - prediction_start

                # If this is a real step and the env is done, reset
                if msg_type == MsgType.STEP_REAL or msg_type == MsgType.STEP_REAL_RESET:
                    for i, result in enumerate(results):
                        obs, reward, done, info = result[0]

                        if self.is_multiagent and all(done):
                            is_done = True
                        elif not self.is_multiagent and done:
                            is_done = True
                        else:
                            is_done = False

                        if is_done or reset[i]:
                            obs = real_envs[i].reset()
                            if not self.is_multiagent:
                                info = self._get_info(real_envs[i])  # info for the new episode

                        results[i] = (obs, reward, done, info)

            self.result_queue.put(results)
            self.task_queue.task_done()


class MultiEnv:
    """Run multiple gym-compatible environments in parallel, keeping more or less the same interface."""

    def __init__(self, num_envs, num_workers, make_env_func, stats_episodes, use_multiprocessing=True):
        self._verbose = False

        if num_workers > num_envs or num_envs % num_workers != 0:
            raise Exception('num_envs should be a multiple of num_workers')

        # create a temp env to query information
        env = make_env_func(None)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        env.close()
        del env

        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers = []

        envs = np.split(np.arange(num_envs), num_workers)
        self.workers = [
            _MultiEnvWorker(envs[i].tolist(), make_env_func, use_multiprocessing)
            for i in range(num_workers)
        ]

        for worker in self.workers:
            worker.task_queue.put((None, MsgType.INIT))
            time.sleep(0.1)  # just in case
        for worker in self.workers:
            worker.task_queue.join()
        log.info('Envs initialized!')

        self.curr_episode_reward = [0] * self._num_actors()
        self.episode_rewards = [[] for _ in range(self._num_actors())]

        self.curr_episode_duration = [0] * self._num_actors()
        self.episode_lengths = [[] for _ in range(self._num_actors())]

        self.stats_episodes = stats_episodes

    def _num_actors(self):
        return self.num_envs

    def _preprocess_data(self, data):
        if data is None:
            data = [None] * self.num_envs

        assert len(data) == self.num_envs
        data = np.split(np.array(data), self.num_workers)
        assert len(data) == self.num_workers
        return data

    def _process_results(self, task_type, timeout):
        results = []
        for worker in self.workers:
            worker.task_queue.join()
            results_per_worker = safe_get(
                worker.result_queue,
                timeout=timeout,
                msg=f'Takes a surprisingly long time to process task {task_type}, retry...',
            )

            assert len(results_per_worker) == self.num_envs // self.num_workers
            results.extend(results_per_worker)
            worker.result_queue.task_done()

        return results

    def await_tasks(self, data, task_type, timeout=None):
        data = self._preprocess_data(data)

        for worker, task in zip(self.workers, data):
            worker.task_queue.put((task, task_type))

        num_actors_per_worker = self._num_actors() // self.num_workers
        if timeout is None:
            timeout = num_actors_per_worker * 0.02

        results = self._process_results(task_type, timeout)
        return results

    def info(self):
        infos = self.await_tasks(None, MsgType.INFO)
        return infos

    def reset(self):
        observations = self.await_tasks(None, MsgType.RESET)
        return observations

    def step(self, actions, reset=None):
        """
        Obviously, returns vectors of obs, rewards, dones instead of usual single values.
        Must call reset before the first step!
        """
        if reset is None:
            results = self.await_tasks(actions, MsgType.STEP_REAL)
        else:
            results = self.await_tasks(list(zip(actions, reset)), MsgType.STEP_REAL_RESET)
        observations, rewards, dones, infos = zip(*results)

        self._update_stats(rewards, dones, infos)
        return observations, rewards, dones, infos

    def _update_stats(self, rewards, dones, infos):
        for i in range(self._num_actors()):
            self.curr_episode_reward[i] += rewards[i]

            step_len = 1
            if infos[i] is not None and 'num_frames' in infos[i]:
                step_len = infos[i]['num_frames']

            self.curr_episode_duration[i] += step_len

            if dones[i]:
                self._update_episode_stats(self.episode_rewards[i], self.curr_episode_reward[i])
                self.curr_episode_reward[i] = 0
                self._update_episode_stats(self.episode_lengths[i], self.curr_episode_duration[i])
                self.curr_episode_duration[i] = 0

    def predict(self, imagined_action_lists):
        start = time.time()
        assert len(imagined_action_lists) == self.num_envs
        imagined_action_lists = np.split(np.array(imagined_action_lists), self.num_workers)
        for worker, imagined_action_list in zip(self.workers, imagined_action_lists):
            worker.task_queue.put((imagined_action_list, MsgType.STEP_IMAGINED))

        observations = []
        rewards = []
        dones = []
        for worker in self.workers:
            worker.task_queue.join()
            results_per_worker = safe_get(
                worker.result_queue,
                timeout=1.0,
                msg='Took a surprisingly long time to predict the future, retrying...',
            )

            assert len(results_per_worker) == len(imagined_action_lists[0])
            for result in results_per_worker:
                o, r, d, _ = zip(*result)
                observations.append(o)
                rewards.append(r)
                dones.append(d)
            worker.result_queue.task_done()

        if self._verbose:
            log.debug('Prediction step took %.4f s', time.time() - start)
        return observations, rewards, dones

    def close(self):
        log.info('Stopping multi env wrapper...')

        for worker in self.workers:
            worker.task_queue.put((None, MsgType.TERMINATE))
            time.sleep(0.1)
        for worker in self.workers:
            worker.process.join()

    def _update_episode_stats(self, episode_stats, curr_episode_data):
        episode_stats_target_size = 2 * (1 + self.stats_episodes // self._num_actors())
        episode_stats.append(curr_episode_data)
        if len(episode_stats) > episode_stats_target_size * 2:
            del episode_stats[:episode_stats_target_size]

    def _calc_episode_stats(self, episode_data, n):
        n_per_actor = 1 + n // self._num_actors()  # number of last episodes to use from every environment

        avg_value = 0
        mean_of_n_episodes = 0
        for i in range(self._num_actors()):
            last_values = episode_data[i][-n_per_actor:]
            if len(last_values) <= 0:
                continue

            avg_value += np.sum(last_values)
            mean_of_n_episodes += len(last_values)

        # to prevent reporting statistics too early
        if mean_of_n_episodes < max(n, self._num_actors()):
            return np.nan

        return avg_value / mean_of_n_episodes

    def calc_avg_rewards(self, n):
        return self._calc_episode_stats(self.episode_rewards, n)

    def calc_avg_episode_lengths(self, n):
        return self._calc_episode_stats(self.episode_lengths, n)

    def stats_num_episodes(self):
        worker_lengths = [len(r) for r in self.episode_rewards]
        return sum(worker_lengths)
