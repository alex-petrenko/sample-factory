import threading
import time
from enum import Enum
from multiprocessing import Process, JoinableQueue
from queue import Empty, Queue

import cv2
import numpy as np

from algorithms.utils.multi_env import MultiEnv, MsgType
from envs.doom.doom_render import concat_grid, cvt_doom_obs
from envs.doom.multiplayer.doom_multiagent import find_available_port, DEFAULT_UDP_PORT
from utils.utils import log, kill


def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.warning(msg)


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, STEP_UPDATE, INFO = range(6)


def init_multiplayer_env(make_env_func, player_id, env_config, init_info=None):
    env = make_env_func(player_id=player_id)

    if env_config is not None and 'worker_index' in env_config:
        env.unwrapped.worker_index = env_config.worker_index
    if env_config is not None and 'vector_index' in env_config:
        env.unwrapped.vector_index = env_config.vector_index

    if init_info is not None:
        env.unwrapped.init_info = init_info

    env.seed(env.unwrapped.worker_index * 1000 + env.unwrapped.vector_index * 10 + player_id)
    return env


class MultiAgentEnvWorker:
    def __init__(self, player_id, make_env_func, env_config, use_multiprocessing=False):
        self.player_id = player_id
        self.make_env_func = make_env_func
        self.env_config = env_config

        if use_multiprocessing:
            self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
            self.process = Process(target=self.start, daemon=False)
        else:
            self.task_queue, self.result_queue = Queue(), Queue()
            self.process = threading.Thread(target=self.start)

        self.process.start()

    def _init(self, init_info):
        log.info('Initializing env for player %d, init_info: %r...', self.player_id, init_info)
        env = init_multiplayer_env(self.make_env_func, self.player_id, self.env_config, init_info)
        env.reset()
        return env

    def _terminate(self, env):
        if env is None:
            return

        # log.info('Stop env for player %d...', self.player_id)
        env.close()
        # log.info('Env with player %d terminated!', self.player_id)

    @staticmethod
    def _get_info(env):
        """Specific to custom VizDoom environments."""
        info = {}
        if hasattr(env.unwrapped, 'get_info_all'):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def start(self):
        env = None

        while True:
            data, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init(data)
                self.result_queue.put(None)  # signal we're done
                self.task_queue.task_done()
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate(env)
                self.task_queue.task_done()
                break

            if task_type == TaskType.RESET:
                results = env.reset()
            elif task_type == TaskType.INFO:
                results = self._get_info(env)
            elif task_type == TaskType.STEP or task_type == TaskType.STEP_UPDATE:
                # collect obs, reward, done, and info
                action = data
                env.unwrapped.update_state = task_type == TaskType.STEP_UPDATE
                results = env.step(action)
            else:
                raise Exception(f'Unknown task type {task_type}')

            self.result_queue.put(results)
            self.task_queue.task_done()


class MultiAgentEnv:
    def __init__(self, num_agents, make_env_func, env_config, skip_frames):
        self.num_agents = num_agents
        log.debug('Multi agent env, num agents: %d', self.num_agents)
        self.skip_frames = skip_frames  # number of frames to skip (1 = no skip)

        env = make_env_func(player_id=-1)  # temporary env just to query observation_space and stuff
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        env.close()

        self.make_env_func = make_env_func

        self.safe_init = env_config is not None and env_config.get('safe_init', True)
        self.safe_init = False  # override

        if self.safe_init:
            sleep_seconds = env_config.worker_index * 1.0
            log.info('Sleeping %.3f seconds to avoid creating all envs at once', sleep_seconds)
            time.sleep(sleep_seconds)
            log.info('Done sleeping at %d', env_config.worker_index)

        self.env_config = env_config
        self.workers = None

        # only needed when rendering
        self.enable_rendering = False
        self.last_obs = None

        self.initialized = False

    def await_tasks(self, data, task_type, timeout=None):
        """
        Task result is always a tuple of lists, e.g.:
        (
            [0th_agent_obs, 1st_agent_obs, ... ],
            [0th_agent_reward, 1st_agent_reward, ... ],
            ...
        )

        If your "task" returns only one result per agent (e.g. reset() returns only the observation),
        the result will be a tuple of lenght 1. It is a responsibility of the caller to index appropriately.

        """
        if data is None:
            data = [None] * self.num_agents

        assert len(data) == self.num_agents

        for i, worker in enumerate(self.workers):
            worker.task_queue.put((data[i], task_type))

        result_lists = None
        for i, worker in enumerate(self.workers):
            results = safe_get(
                worker.result_queue,
                timeout=0.2 if timeout is None else timeout,
                msg=f'Takes a surprisingly long time to process task {task_type}, retry...',
            )

            worker.result_queue.task_done()
            worker.task_queue.join()

            if not isinstance(results, (tuple, list)):
                results = [results]

            if result_lists is None:
                result_lists = tuple([] for _ in results)

            for j, r in enumerate(results):
                result_lists[j].append(r)

        return result_lists

    def _ensure_initialized(self):
        if self.initialized:
            return

        num_attempts = 25
        attempt = 0
        for attempt in range(num_attempts):
            self.workers = [
                MultiAgentEnvWorker(i, self.make_env_func, self.env_config) for i in range(self.num_agents)
            ]

            try:
                port_to_use = DEFAULT_UDP_PORT + 100 * self.env_config.worker_index + self.env_config.vector_index
                port = find_available_port(port_to_use, increment=1000)
                log.debug('Using port %d', port)
                init_info = dict(port=port)

                for i, worker in enumerate(self.workers):
                    worker.task_queue.put((init_info, TaskType.INIT))
                    if self.safe_init:
                        time.sleep(1.0)  # just in case
                    else:
                        time.sleep(0.01)

                for i, worker in enumerate(self.workers):
                    worker.result_queue.get(timeout=5)
                    worker.result_queue.task_done()
                    worker.task_queue.join()
            except Exception as exc:
                for worker in self.workers:
                    if isinstance(worker.process, threading.Thread):
                        log.info('We cannot really kill a thread, so let the whole process die')
                        raise RuntimeError('Critical error: worker stuck on initialization. Abort!')
                    else:
                        log.info('Killing process %r', worker.process.pid)
                        kill(worker.process.pid)
                del self.workers
                log.warning('Could not initialize env, try again! Error: %r', exc)
                time.sleep(1)
            else:
                break

        if attempt >= num_attempts:
            log.error('Could not initialize env even after %d attempts. Fail!', attempt)
            raise RuntimeError('Critical error: worker stuck on initialization, num attempts exceeded. Abort!')

        log.debug('%d agent workers initialized for env %d!', len(self.workers), self.env_config.worker_index)
        log.debug('Took %d attempts!\n', attempt + 1)
        self.initialized = True

    def info(self):
        self._ensure_initialized()
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    def reset(self):
        self._ensure_initialized()
        observation = self.await_tasks(None, TaskType.RESET, timeout=2.0)[0]
        return observation

    def step(self, actions):
        self._ensure_initialized()

        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.STEP)
        obs, rew, dones, infos = self.await_tasks(actions, TaskType.STEP_UPDATE)
        for info in infos:
            info['num_frames'] = self.skip_frames

        if all(dones):
            obs = self.await_tasks(None, TaskType.RESET, timeout=2.0)[0]

        if self.enable_rendering:
            self.last_obs = obs

        return obs, rew, dones, infos

    # noinspection PyUnusedLocal
    def render(self, *args, **kwargs):
        self.enable_rendering = True

        if self.last_obs is None:
            return

        render_multiagent = True
        if render_multiagent:
            obs_display = [o['obs'] for o in self.last_obs]
            obs_grid = concat_grid(obs_display)
            cv2.imshow('vizdoom', obs_grid)
        else:
            obs_display = self.last_obs[0]['obs']
            cv2.imshow('vizdoom', cvt_doom_obs(obs_display))

        cv2.waitKey(1)

    def close(self):
        if self.workers is not None:
            # log.info('Stopping multiagent env %d...', self.env_config.worker_index)
            for worker in self.workers:
                worker.task_queue.put((None, TaskType.TERMINATE))
                time.sleep(0.1)
            for worker in self.workers:
                worker.process.join()

    def seed(self, seed=None):
        """Does not really make sense for the wrapper. Individual envs will be uniquely seeded on init."""
        pass


class MultiAgentEnvAggregator(MultiEnv):
    """
    Vectorized wrapper for multi-agent envs. This is for a special usecase where all agents (policies) are the same,
    and therefore each agent in a multi-env can be treated as a separate env.
    """
    def __init__(self, num_envs, num_workers, make_env_func, stats_episodes, use_multiprocessing=True):
        tmp_env = make_env_func(None)
        if hasattr(tmp_env, 'num_agents'):
            self.num_agents = tmp_env.num_agents
        else:
            raise Exception('Expected multi-agent environment')

        global DEFAULT_UDP_PORT
        DEFAULT_UDP_PORT = find_available_port(DEFAULT_UDP_PORT)
        log.debug('Default UDP port changed to %r', DEFAULT_UDP_PORT)
        time.sleep(0.1)

        super().__init__(num_envs, num_workers, make_env_func, stats_episodes, use_multiprocessing)

    def _num_actors(self):
        return self.num_envs * self.num_agents

    def _preprocess_data(self, data):
        """Each multi-agent environment expects a dict, with one action per agent."""
        if data is None:
            data = [None] * self.num_agents * self.num_envs

        assert len(data) == self.num_agents * self.num_envs
        data = np.array(data)
        data = np.reshape(data, [self.num_envs, self.num_agents] + list(data.shape[1:]))
        data = np.split(data, self.num_workers)
        assert len(data) == self.num_workers
        return data

    def step(self, actions, reset=None):
        if reset is None:
            results = self.await_tasks(actions, MsgType.STEP_REAL)
        else:
            results = self.await_tasks(list(zip(actions, reset)), MsgType.STEP_REAL_RESET)

        log.info('After await tasks')
        observations, rewards, dones, infos = zip(*results)

        self._update_stats(rewards, dones, infos)
        return observations, rewards, dones, infos
