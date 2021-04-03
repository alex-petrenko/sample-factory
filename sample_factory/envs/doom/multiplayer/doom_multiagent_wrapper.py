import threading
import time
from enum import Enum
from multiprocessing import Process
from queue import Empty, Queue
import faster_fifo

import cv2
import filelock
import gym
from filelock import FileLock

from sample_factory.envs.doom.doom_gym import doom_lock_file
from sample_factory.envs.doom.doom_render import concat_grid, cvt_doom_obs
from sample_factory.envs.doom.multiplayer.doom_multiagent import find_available_port, DEFAULT_UDP_PORT
from sample_factory.envs.env_utils import RewardShapingInterface, get_default_reward_shaping
from sample_factory.utils.utils import log
from functools import wraps
from time import sleep


def retry_dm(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    # This accesses the self instance variable
                    multiagent_wrapper_obj = args[0]
                    multiagent_wrapper_obj.initialized = False
                    multiagent_wrapper_obj.close()

                    # This is done to reset if it is in the step function
                    if should_reset:
                        multiagent_wrapper_obj.reset()

                    if i == num_attempts - 1:
                        raise
                    else:
                        log.error('Failed with error %r, trying again', e)
                        sleep(sleep_time)

        return wrapper
    return decorator


def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.warning(msg)


def udp_port_num(env_config):
    if env_config is None:
        return DEFAULT_UDP_PORT
    port_to_use = DEFAULT_UDP_PORT + 100 * env_config.worker_index + env_config.vector_index
    return port_to_use


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, STEP_UPDATE, INFO, SET_ATTR = range(7)


def init_multiplayer_env(make_env_func, player_id, env_config, init_info=None):
    env = make_env_func(player_id=player_id)

    if env_config is not None and 'worker_index' in env_config:
        env.unwrapped.worker_index = env_config.worker_index
    if env_config is not None and 'vector_index' in env_config:
        env.unwrapped.vector_index = env_config.vector_index

    if init_info is None:
        port_to_use = udp_port_num(env_config)
        port = find_available_port(port_to_use, increment=1000)
        log.debug('Using port %d', port)
        init_info = dict(port=port)

    env.unwrapped.init_info = init_info

    env.seed(env.unwrapped.worker_index * 1000 + env.unwrapped.vector_index * 10 + player_id)
    return env


class MultiAgentEnvWorker:
    def __init__(self, player_id, make_env_func, env_config, use_multiprocessing=False, reset_on_init=True):
        self.player_id = player_id
        self.make_env_func = make_env_func
        self.env_config = env_config
        self.reset_on_init = reset_on_init
        if use_multiprocessing:
            self.process = Process(target=self.start, daemon=False)
            self.task_queue, self.result_queue = faster_fifo.Queue(), faster_fifo.Queue()
        else:
            self.process = threading.Thread(target=self.start)
            self.task_queue, self.result_queue = Queue(), Queue()

        self.process.start()

    def _init(self, init_info):
        log.info('Initializing env for player %d, init_info: %r...', self.player_id, init_info)
        env = init_multiplayer_env(self.make_env_func, self.player_id, self.env_config, init_info)
        if self.reset_on_init:
            env.reset()
        return env

    @staticmethod
    def _terminate(env):
        if env is None:
            return
        env.close()

    @staticmethod
    def _get_info(env):
        """Specific to custom VizDoom environments."""
        info = {}
        if hasattr(env.unwrapped, 'get_info_all'):
            info = env.unwrapped.get_info_all()  # info for the new episode
        return info

    def _set_env_attr(self, env, player_id, attr_chain, value):
        """Allows us to set an arbitrary attribute of the environment, e.g. attr_chain can be unwrapped.foo.bar"""
        assert player_id == self.player_id

        attrs = attr_chain.split('.')
        curr_attr = env
        try:
            for attr_name in attrs[:-1]:
                curr_attr = getattr(curr_attr, attr_name)
        except AttributeError:
            log.error('Env does not have an attribute %s', attr_chain)

        attr_to_set = attrs[-1]
        setattr(curr_attr, attr_to_set, value)

    def start(self):
        env = None

        while True:
            data, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init(data)
                self.result_queue.put(None)  # signal we're done
                continue

            if task_type == TaskType.TERMINATE:
                self._terminate(env)
                break

            results = None
            if task_type == TaskType.RESET:
                results = env.reset()
            elif task_type == TaskType.INFO:
                results = self._get_info(env)
            elif task_type == TaskType.STEP or task_type == TaskType.STEP_UPDATE:
                # collect obs, reward, done, and info
                action = data
                env.unwrapped.update_state = task_type == TaskType.STEP_UPDATE
                results = env.step(action)
            elif task_type == TaskType.SET_ATTR:
                player_id, attr_chain, value = data
                self._set_env_attr(env, player_id, attr_chain, value)
            else:
                raise Exception(f'Unknown task type {task_type}')

            self.result_queue.put(results)


class MultiAgentEnv(gym.Env, RewardShapingInterface):
    def __init__(self, num_agents, make_env_func, env_config, skip_frames):
        gym.Env.__init__(self)
        RewardShapingInterface.__init__(self)

        self.num_agents = num_agents
        log.debug('Multi agent env, num agents: %d', self.num_agents)
        self.skip_frames = skip_frames  # number of frames to skip (1 = no skip)

        env = make_env_func(player_id=-1)  # temporary env just to query observation_space and stuff
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.default_reward_shaping = get_default_reward_shaping(env)
        env.close()

        self.current_reward_shaping = [self.default_reward_shaping for _ in range(self.num_agents)]

        self.make_env_func = make_env_func

        self.safe_init = env_config is not None and env_config.get('safe_init', False)

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

        self.reset_on_init = True

        self.initialized = False

    def get_default_reward_shaping(self):
        return self.default_reward_shaping

    def get_current_reward_shaping(self, agent_idx: int):
        return self.current_reward_shaping[agent_idx]

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        self.current_reward_shaping[agent_idx] = reward_shaping
        self.set_env_attr(
            agent_idx, 'unwrapped.reward_shaping_interface.reward_shaping_scheme', reward_shaping,
        )

    def await_tasks(self, data, task_type, timeout=None):
        """
        Task result is always a tuple of lists, e.g.:
        (
            [0th_agent_obs, 1st_agent_obs, ... ],
            [0th_agent_reward, 1st_agent_reward, ... ],
            ...
        )

        If your "task" returns only one result per agent (e.g. reset() returns only the observation),
        the result will be a tuple of length 1. It is a responsibility of the caller to index appropriately.

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

        self.workers = [
            MultiAgentEnvWorker(i, self.make_env_func, self.env_config, reset_on_init=self.reset_on_init)
            for i in range(self.num_agents)
        ]

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                port_to_use = udp_port_num(self.env_config)
                port = find_available_port(port_to_use, increment=1000)
                log.debug('Using port %d', port)
                init_info = dict(port=port)

                lock_file = doom_lock_file(max_parallel=20)
                lock = FileLock(lock_file)
                with lock.acquire(timeout=10):
                    for i, worker in enumerate(self.workers):
                        worker.task_queue.put((init_info, TaskType.INIT))
                        if self.safe_init:
                            time.sleep(1.0)  # just in case
                        else:
                            time.sleep(0.05)

                    for i, worker in enumerate(self.workers):
                        worker.result_queue.get(timeout=20)

            except filelock.Timeout:
                continue
            except Exception:
                raise RuntimeError('Critical error: worker stuck on initialization. Abort!')
            else:
                break

        log.debug('%d agent workers initialized for env %d!', len(self.workers), self.env_config.worker_index)
        self.initialized = True

    @retry_dm(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False)
    def info(self):
        self._ensure_initialized()
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    @retry_dm(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=False)
    def reset(self):
        self._ensure_initialized()
        observation = self.await_tasks(None, TaskType.RESET, timeout=2.0)[0]
        return observation

    @retry_dm(exception_class=Exception, num_attempts=3, sleep_time=1, should_reset=True)
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

    def set_env_attr(self, agent_idx, attr_chain, value):
        data = (agent_idx, attr_chain, value)
        worker = self.workers[agent_idx]
        worker.task_queue.put((data, TaskType.SET_ATTR))

        result = safe_get(worker.result_queue, timeout=0.1)
        assert result is None
