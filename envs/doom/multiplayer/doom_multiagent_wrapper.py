import time
from enum import Enum
from multiprocessing import Process, JoinableQueue
from queue import Empty

import cv2
from ray.rllib import MultiAgentEnv

from envs.doom.doom_render import concat_grid, cvt_doom_obs
from utils.utils import log


def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.exception(msg)


class TaskType(Enum):
    INIT, TERMINATE, RESET, STEP, STEP_UPDATE, INFO = range(6)


def init_multiplayer_env(make_env_func, player_id, env_config):
    env = make_env_func(player_id=player_id)

    if env_config is not None and 'worker_index' in env_config:
        env.unwrapped.worker_index = env_config.worker_index
    if env_config is not None and 'vector_index' in env_config:
        env.unwrapped.vector_index = env_config.vector_index

    env.seed(env.unwrapped.worker_index * 1000 + env.unwrapped.vector_index * 10 + player_id)
    return env


class MultiAgentEnvWorker:
    def __init__(self, player_id, make_env_func, env_config):
        self.player_id = player_id
        self.make_env_func = make_env_func
        self.env_config = env_config

        self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
        self.process = Process(target=self.start, daemon=True)
        self.process.start()

    def _init(self):
        log.info('Initializing env for player %d...', self.player_id)
        env = init_multiplayer_env(self.make_env_func, self.player_id, self.env_config)
        return env

    def _terminate(self, env):
        if env is None:
            return

        log.info('Stop env for player %d...', self.player_id)
        env.close()
        log.info('Env with player %d terminated!', self.player_id)

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
            action, task_type = safe_get(self.task_queue)

            if task_type == TaskType.INIT:
                env = self._init()
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
                env.unwrapped.update_state = task_type == TaskType.STEP_UPDATE
                results = env.step(action)
            else:
                raise Exception(f'Unknown task type {task_type}')

            self.result_queue.put(results)
            self.task_queue.task_done()


class VizdoomMultiAgentEnv(MultiAgentEnv):
    def __init__(self, num_agents, make_env_func, env_config, skip_frames):
        self.num_agents = num_agents
        self.skip_frames = skip_frames  # number of frames to skip (1 = no skip)

        env = make_env_func(player_id=-1)  # temporary env just to query observation_space and stuff
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        env.close()

        self.safe_init = env_config is not None and env_config.get('safe_init', True)
        self.safe_init = False  # override

        if self.safe_init:
            sleep_seconds = env_config.worker_index * 1.0
            log.info('Sleeping %.3f seconds to avoid creating all envs at once', sleep_seconds)
            time.sleep(sleep_seconds)
            log.info('Done sleeping at %d', env_config.worker_index)

        self.workers = [MultiAgentEnvWorker(i, make_env_func, env_config) for i in range(num_agents)]

        # only needed when rendering
        self.enable_rendering = False
        self.last_obs = None

        self.initialized = False

    def await_tasks(self, data, task_type, timeout=None):
        """
        Task result is always a tuple of dicts, e.g.:
        (
            {'0': 0th_agent_obs, '1': 1st_agent_obs, ... ,
            {'0': 0th_agent_reward, '1': 1st_agent_obs, ... ,
            ...
        )

        If your "task" returns only one result per agent (e.g. reset() returns only the observation),
        the result will be a tuple of lenght 1. It is a responsibility of the caller to index appropriately.

        """
        if data is None:
            data = {str(i): None for i in range(self.num_agents)}

        assert len(data) == self.num_agents

        for i, worker in enumerate(self.workers[1:], start=1):
            worker.task_queue.put((data[str(i)], task_type))
        self.workers[0].task_queue.put((data[str(0)], task_type))

        result_dicts = None
        for i, worker in enumerate(self.workers):
            worker.task_queue.join()
            results = safe_get(
                worker.result_queue,
                timeout=0.04 if timeout is None else timeout,
                msg=f'Takes a surprisingly long time to process task {task_type}, retry...',
            )

            worker.result_queue.task_done()

            if not isinstance(results, (tuple, list)):
                results = [results]

            if result_dicts is None:
                result_dicts = tuple({} for _ in results)

            for j, r in enumerate(results):
                result_dicts[j][str(i)] = r

        return result_dicts

    def _ensure_initialized(self):
        if self.initialized:
            return

        for worker in self.workers:
            worker.task_queue.put((None, TaskType.INIT))
            if self.safe_init:
                time.sleep(1.0)  # just in case
            else:
                time.sleep(0.25)

        for worker in self.workers:
            worker.task_queue.join()

        log.info('%d agent workers initialized!', len(self.workers))
        self.initialized = True

    def info(self):
        self._ensure_initialized()
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    def reset(self):
        self._ensure_initialized()
        observation = self.await_tasks(None, TaskType.RESET)[0]
        return observation

    def step(self, actions):
        self._ensure_initialized()

        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.STEP)
        obs, rew, dones, infos = self.await_tasks(actions, TaskType.STEP_UPDATE)
        dones['__all__'] = all(dones.values())

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
            obs_display = [o['obs'] for o in self.last_obs.values()]
            obs_grid = concat_grid(obs_display)
            cv2.imshow('vizdoom', obs_grid)
        else:
            obs_display = self.last_obs['0']['obs']
            cv2.imshow('vizdoom', cvt_doom_obs(obs_display))

        cv2.waitKey(1)

    def close(self):
        log.info('Stopping multi env...')

        for worker in self.workers:
            worker.task_queue.put((None, TaskType.TERMINATE))
            time.sleep(0.1)
        for worker in self.workers:
            worker.process.join()
