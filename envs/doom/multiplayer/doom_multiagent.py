import copy
import time
from enum import Enum
from multiprocessing import Process, JoinableQueue
from queue import Empty

import cv2
import numpy as np
from ray.rllib import MultiAgentEnv
from vizdoom import *

from envs.doom.doom_gym import VizdoomEnv
from envs.doom.doom_helpers import concat_grid, cvt_doom_obs
from utils.utils import log


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(
            self,
            action_space,
            config_file,
            player_id, num_agents, max_num_players, num_bots,
            skip_frames, async_mode=False,
    ):
        super().__init__(action_space, config_file, skip_frames=skip_frames, async_mode=async_mode)

        self.worker_index = 0
        self.vector_index = 0

        self.player_id = player_id
        self.num_agents = num_agents  # num agents that are not humans or bots
        self.max_num_players = max_num_players
        self.max_num_bots = num_bots
        self.min_num_bots = min(8, self.max_num_bots)
        self.timestep = 0
        self.update_state = True

        # hardcode bot names for consistency, otherwise they are generated randomly
        self.bot_names = [
            'Blazkowicz',
            'PerfectBlue',
            'PerfectRed',
            'PerfectGreen',
            'PerfectPurple',
            'PerfectYellow',
            'PerfectWhite',
            'PerfectLtGreen',
        ]
        self.bot_difficulty_mean = self.bot_difficulty_std = None
        self.hardest_bot = 100
        self.easiest_bot = 10

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        self.game.set_seed(self.rng.randint(0, 2**32-1))

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human' and self._is_server():
            self.game.set_window_visible(True)

        # make sure not to use more than 10 envs per worker
        port = 50300 + self.worker_index * 100 + self.vector_index + 7
        log.info('Using port %d...', port)

        if self._is_server():
            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            self.game.add_game_args(
                f'-host {self.max_num_players} -port {port} '
                '-deathmatch '  # Deathmatch rules are used for the game.                
                '+timelimit 4.0 '  # The game (episode) will end after this many minutes have elapsed.
                '+sv_forcerespawn 1 '  # Players will respawn automatically after they die.
                '+sv_noautoaim 1 '  # Autoaim is disabled for all players.
                '+sv_respawnprotect 1 '  # Players will be invulnerable for two second after spawning.
                '+sv_spawnfarthest 1 '  # Players will be spawned as far as possible from any other players.
                '+sv_nocrouch 1 '  # Disables crouching.
                '+viz_respawn_delay 0 '  # Sets delay between respanws (in seconds).
                '+viz_connect_timeout 999 '  # In seconds
            )

            # Additional commands:
            #
            # disables depth and labels buffer and the ability to use commands
            # that could interfere with multiplayer game (should use this in evaluation)
            # '+viz_nocheat 1'

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f'+name AI{self.player_id}_host +colorset 0')
        else:
            # Join existing game.
            self.game.add_game_args(
                f'-join 127.0.0.1:{port} '  # Connect to a host for a multiplayer game.
                '+viz_connect_timeout 999 '
            )

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f'+name AI{self.player_id} +colorset 0')

        self._set_game_mode()
        self.game.init()

        log.info('Initialized w:%d v:%d', self.worker_index, self.vector_index)
        self.initialized = True

    def _random_bot(self, difficulty, used_bots):
        while True:
            idx = self.rng.randint(0, self.max_num_bots)
            bot_name = f'BOT_{difficulty}_{idx}'
            if bot_name not in used_bots:
                used_bots.append(bot_name)
                return bot_name

    def reset(self, mode='algo'):
        obs = super().reset(mode)

        if self._is_server() and self.max_num_bots > 0:
            self.game.send_game_command('removebots')
            if self.launch_mode == 'test':
                num_bots = self.max_num_bots
            else:
                num_bots = self.rng.randint(self.min_num_bots, self.max_num_bots + 1)

            bot_names = copy.deepcopy(self.bot_names)
            self.rng.shuffle(bot_names)

            used_bots = []

            for i in range(num_bots):
                if self.bot_difficulty_mean is None:
                    # add named bots from the list

                    if i < len(bot_names):
                        bot_name = ' ' + bot_names[i]
                    else:
                        bot_name = ''

                    log.info('Adding bot %d %s', i, bot_name)
                    self.game.send_game_command(f'addbot{bot_name}')
                else:
                    # add random bots according to the desired difficulty
                    diff = self.rng.normal(self.bot_difficulty_mean, self.bot_difficulty_std)
                    diff = int(round(diff, -1))
                    diff = max(self.easiest_bot, diff)
                    diff = min(self.hardest_bot, diff)
                    bot_name = self._random_bot(diff, used_bots)
                    log.info('Adding bot %d %s', i, bot_name)
                    self.game.send_game_command(f'addbot {bot_name}')

        self.timestep = 0
        self.update_state = True
        return obs

    def step(self, actions):
        if self.skip_frames > 1 or self.num_agents == 1:
            # not used in multi-agent mode due to VizDoom limitations
            # this means that we have only one agent (+ maybe some bots, which is why we're in multiplayer mode)
            return super().step(actions)

        self._ensure_initialized()
        info = {}

        actions_binary = self._convert_actions(actions)

        self.game.set_action(actions_binary)
        self.game.advance_action(1, self.update_state)
        self.timestep += 1

        if not self.update_state:
            return None, None, None, None

        state = self.game.get_state()
        reward = self.game.get_last_reward()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        self._vizdoom_variables_bug_workaround(info, done)
        return observation, reward, done, info


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

    if env_config is not None:
        env.unwrapped.worker_index = env_config.worker_index
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
