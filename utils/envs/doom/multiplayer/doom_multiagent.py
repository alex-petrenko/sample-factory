import time
from enum import Enum
from multiprocessing import Process, JoinableQueue
from queue import Empty
from random import choice

import numpy as np
from vizdoom import *
from vizdoomgym.envs import VizdoomEnv

from utils.utils import log


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(self, level, player_id, num_players, skip_frames, level_map='map01'):
        super().__init__(level, skip_frames=skip_frames, level_map=level_map)

        self.worker_index = 0
        self.vector_index = 0

        self.player_id = player_id
        self.num_players = num_players
        self.timestep = 0
        self.update_state = True

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        # Setting an invalid level map will cause the game to freeze silently
        self.game.set_doom_map(self.level_map)
        self.game.set_seed(self.rng.random_integers(0, 2**32-1))

        if mode == 'algo':
            self.game.set_window_visible(False)

        # make sure not to use more than 10 envs per worker
        port = 50300 + self.worker_index * 10 + self.vector_index
        log.info('Using port %d...', port)

        if self._is_server():
            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            self.game.add_game_args(
                f'-host {self.num_players} -port {port} '
                '-deathmatch '  # Deathmatch rules are used for the game.                
                '+timelimit 10.0 '  # The game (episode) will end after this many minutes have elapsed.
                '+sv_forcerespawn 1 '  # Players will respawn automatically after they die.
                '+sv_noautoaim 1 '  # Autoaim is disabled for all players.
                '+sv_respawnprotect 1 '  # Players will be invulnerable for two second after spawning.
                '+sv_spawnfarthest 1 '  # Players will be spawned as far as possible from any other players.
                '+sv_nocrouch 1 '  # Disables crouching.
                '+viz_respawn_delay 1 '  # Sets delay between respanws (in seconds).
                '+viz_nocheat 1',  # Disables depth and labels buffer and the ability to use commands
                                   # that could interfere with multiplayer game.
            )

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args('+name Host +colorset 0')
        else:
            # TODO: port, name
            # Join existing game.
            self.game.add_game_args(f'-join 127.0.0.1:{port}')  # Connect to a host for a multiplayer game.

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args('+name AI +colorset 0')

        self.game.set_mode(Mode.PLAYER)

        self.game.init()

        self.initialized = True

    def reset(self, mode='algo'):
        self._ensure_initialized(mode)
        self.timestep = 0
        self.update_state = True
        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer
        return np.transpose(img, (1, 2, 0))

    def step(self, action):
        self._ensure_initialized()
        info = {'num_frames': self.skip_frames}

        # convert action to vizdoom action space (one hot)
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()

        reward = 0

        self.game.set_action(act)
        self.game.advance_action(1, self.update_state)
        reward += self.game.get_last_reward()
        self.timestep += 1

        if not self.update_state:
            return None, None, None, None

        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

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


class MultiAgentEnvWorker:
    def __init__(self, player_id, num_players, make_env_func, env_config):
        self.player_id = player_id
        self.num_players = num_players
        self.make_env_func = make_env_func
        self.env_config = env_config

        self.task_queue, self.result_queue = JoinableQueue(), JoinableQueue()
        self.process = Process(target=self.start, daemon=True)
        self.process.start()

    def _init(self):
        log.info('Initializing env for player %d...', self.player_id)
        env = self.make_env_func(player_id=self.player_id, num_players=self.num_players)
        env.unwrapped.worker_index = self.env_config['worker_index']
        env.unwrapped.vector_index = self.env_config['vector_index']
        env.seed(self.player_id)
        return env

    def _terminate(self, env):
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


class VizdoomMultiAgentEnv:
    def __init__(self, num_players, make_env_func, env_config):
        self.num_players = num_players
        self.skip_frames = 4

        env = make_env_func(player_id=-1, num_players=num_players)  # temporary
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        env.close()

        self.workers = [MultiAgentEnvWorker(i, num_players, make_env_func, env_config) for i in range(num_players)]

        for worker in self.workers:
            worker.task_queue.put((None, TaskType.INIT))
            time.sleep(0.1)  # just in case
        for worker in self.workers:
            worker.task_queue.join()

        log.info('%d agent workers initialized!', len(self.workers))

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
            data = {str(i): None for i in range(self.num_players)}

        assert len(data) == self.num_players

        for i, worker in enumerate(self.workers[1:], start=1):
            worker.task_queue.put((data[str(i)], task_type))
        self.workers[0].task_queue.put((data[str(0)], task_type))

        result_dicts = None
        for i, worker in enumerate(self.workers):
            worker.task_queue.join()
            results = safe_get(
                worker.result_queue,
                timeout=0.02 if timeout is None else timeout,
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

    def info(self):
        info = self.await_tasks(None, TaskType.INFO)[0]
        return info

    def reset(self):
        observation = self.await_tasks(None, TaskType.RESET)[0]
        return observation

    def step(self, actions):
        for frame in range(self.skip_frames - 1):
            self.await_tasks(actions, TaskType.STEP)
        obs, rew, dones, infos = self.await_tasks(actions, TaskType.STEP_UPDATE)
        dones['__all__'] = all(dones.values())
        return obs, rew, dones, infos

    def close(self):
        log.info('Stopping multi env...')

        for worker in self.workers:
            worker.task_queue.put((None, TaskType.TERMINATE))
            time.sleep(0.1)
        for worker in self.workers:
            worker.process.join()


def start_host(num_players):
    game = DoomGame()

    # Use CIG example config or your own.
    game.load_config("/home/apetrenk/all/projects/doom/vizdoomgym/vizdoomgym/envs/scenarios/cig.cfg")

    game.set_doom_map("map01")  # Limited deathmatch.
    # game.set_doom_map("map02")  # Full deathmatch.

    # Host game with options that will be used in the competition.
    game.add_game_args(f"-host {num_players} -port 5030 "
                       # This machine will function as a host for a multiplayer game with this many players (including this machine).
                       # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
                       "-deathmatch "  # Deathmatch rules are used for the game.
                       "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
                       "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
                       "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
                       "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
                       "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
                       "+sv_nocrouch 1 "  # Disables crouching.
                       "+viz_respawn_delay 1 "  # Sets delay between respanws (in seconds).
                       "+viz_nocheat 1")  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

    # This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
    # game.add_game_args("+viz_spectator 1")

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name Host +colorset 0")

    # During the competition, async mode will be forced for all agents.
    game.set_mode(Mode.PLAYER)
    # game.set_mode(Mode.ASYNC_PLAYER)

    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(False)

    game.init()

    start = time.time()
    num_frames = 10000
    for i in range(num_frames):
        if not advance_player(game, 0, i):
            game.new_episode()

    elapsed = time.time() - start
    fps = num_frames / elapsed
    log.info('FPS on the server: %.1f', fps)

    game.close()


def start_client(client_id):
    game = DoomGame()

    # Use CIG example config or your own.
    game.load_config("/home/apetrenk/all/projects/doom/vizdoomgym/vizdoomgym/envs/scenarios/cig.cfg")

    game.set_doom_map("map01")  # Limited deathmatch.
    # game.set_doom_map("map02")  # Full deathmatch.

    # Join existing game.
    game.add_game_args("-join 127.0.0.1:5030")  # Connect to a host for a multiplayer game.
    # game.add_game_args("-port 5029")  # Connect to a host for a multiplayer game.

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")

    # During the competition, async mode will be forced for all agents.
    game.set_mode(Mode.PLAYER)
    # game.set_mode(Mode.ASYNC_PLAYER)

    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(False)

    game.init()

    time.sleep(0.5)

    start = time.time()
    num_frames = 10000
    for i in range(num_frames):
        if not advance_player(game, client_id, i):
            time.sleep(0.005)
            game.new_episode()

    elapsed = time.time() - start
    fps = num_frames / elapsed
    log.info('FPS on the client: %.1f', fps)


def advance_player(game, client_id, timestep):
    actions = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]

    # Play until the game (episode) is over.
    if not game.is_episode_finished():
        # Get the state.
        s = game.get_state()
        # if client_id != 0:
        #     sleep(random() * 0.5 + 0.001)
        server_state = game.get_server_state()

        player_type = 'host' if client_id == 0 else f'client_{client_id}'
        if client_id == 0 and timestep % 100 == 0:
            log.info('Server state tic %d at %s', server_state.tic, player_type)

        # Analyze the state.

        # Make your action.
        game.make_action(choice(actions), 4)

        # Check if player is dead
        if game.is_player_dead():
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

        return True
    else:
        log.info('Game finished')
        return False


def main():
    num_players = 8

    for i in range(1, num_players):
        log.info('Starting client #%d', i)
        client = Process(target=start_client, args=(i,))
        client.start()

    log.info('Starting host...')
    start_host(num_players)


if __name__ == '__main__':
    main()

