import copy
import os
import random
import re
import time
from os.path import join
from threading import Thread

import cv2
import gym
import numpy as np
from filelock import FileLock, Timeout
from gym.utils import seeding
from vizdoom.vizdoom import ScreenResolution, DoomGame, Mode, AutomapMode

from sample_factory.algorithms.utils.spaces.discretized import Discretized
from sample_factory.utils.utils import log, project_tmp_dir


def doom_lock_file(max_parallel):
    """
    Doom instances tend to have problems starting when a lot of them are initialized in parallel.
    This is not a problem during normal execution once the envs are initialized.

    The "sweet spot" for the number of envs that can be initialized in parallel is about 5-10.
    Here we use file locking mechanism to ensure that only a limited amount of envs are being initialized at the same
    time.
    This tends to be more of a problem for multiplayer envs.

    This also has an advantage of working across completely independent process groups, e.g. different experiments.
    """
    lock_filename = f'doom_{random.randrange(0, max_parallel):03d}.lockfile'

    tmp_dir = project_tmp_dir()
    lock_path = join(tmp_dir, lock_filename)
    return lock_path


def key_to_action_default(key):
    """
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        SELECT_WEAPON1
        SELECT_WEAPON2
        SELECT_WEAPON3
        SELECT_WEAPON4
        SELECT_WEAPON5
        SELECT_WEAPON6
        SELECT_WEAPON7
        ATTACK
        SPEED
        TURN_LEFT_RIGHT_DELTA
    """
    from pynput.keyboard import Key

    # health gathering
    action_table = {
        Key.left: 0,
        Key.right: 1,
        Key.up: 2,
        Key.down: 3,
    }

    # action_table = {
    #     Key.up: 0,
    #     Key.down: 1,
    #     Key.alt: 6,
    #     Key.ctrl: 11,
    #     Key.shift: 12,
    #     Key.space: 13,
    #     Key.right: 'turn_right',
    #     Key.left: 'turn_left',
    # }

    return action_table.get(key, None)


class VizdoomEnv(gym.Env):

    def __init__(self,
                 action_space,
                 config_file,
                 coord_limits=None,
                 max_histogram_length=200,
                 show_automap=False,
                 skip_frames=1,
                 async_mode=False,
                 record_to=None):
        self.initialized = False

        # essential game data
        self.game = None
        self.state = None
        self.curr_seed = 0
        self.rng = None
        self.skip_frames = skip_frames
        self.async_mode = async_mode

        # optional - for topdown view rendering and visitation heatmaps
        self.show_automap = show_automap
        self.coord_limits = coord_limits

        # can be adjusted after the environment is created (but before any reset() call) via observation space wrapper
        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.black_screen = None

        # provided as a part of environment definition, since these depend on the scenario and
        # can be quite complex multi-discrete spaces
        self.action_space = action_space
        self.composite_action_space = hasattr(self.action_space, 'spaces')

        self.delta_actions_scaling_factor = 7.5

        if os.path.isabs(config_file):
            self.config_path = config_file
        else:
            scenarios_dir = join(os.path.dirname(__file__), 'scenarios')
            self.config_path = join(scenarios_dir, config_file)
            if not os.path.isfile(self.config_path):
                log.warning(
                    'File %s not found in scenarios dir %s. Consider providing absolute path?',
                    config_file, scenarios_dir,
                )

        self.variable_indices = self._parse_variable_indices(self.config_path)

        # only created if we call render() method
        self.viewer = None

        # record full episodes using VizDoom recording functionality
        self.record_to = record_to

        self.is_multiplayer = False  # overridden in derived classes

        # (optional) histogram to track positional coverage
        # do not pass coord_limits if you don't need this, to avoid extra calculation
        self.max_histogram_length = max_histogram_length
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            x = (self.coord_limits[2] - self.coord_limits[0])
            y = (self.coord_limits[3] - self.coord_limits[1])
            if x > y:
                len_x = self.max_histogram_length
                len_y = int((y / x) * self.max_histogram_length)
            else:
                len_x = int((x / y) * self.max_histogram_length)
                len_y = self.max_histogram_length
            self.current_histogram = np.zeros((len_x, len_y), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)

        # helpers for human play with pynput keyboard input
        self._terminate = False
        self._current_actions = []
        self._actions_flattened = None

        self._prev_info = None
        self._last_episode_info = None

        self._num_episodes = 0

        self.mode = 'algo'

        self.seed()

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed, max_bytes=4)
        self.rng, _ = seeding.np_random(seed=self.curr_seed)
        return [self.curr_seed, self.rng]

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)

    def _set_game_mode(self, mode):
        if mode == 'replay':
            self.game.set_mode(Mode.PLAYER)
        else:
            if self.async_mode:
                log.info('Starting in async mode! Use this only for testing, otherwise PLAYER mode is much faster')
                self.game.set_mode(Mode.ASYNC_PLAYER)
            else:
                self.game.set_mode(Mode.PLAYER)

    def _create_doom_game(self, mode):
        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        self.game.set_seed(self.rng.randint(0, 2**32 - 1))

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human' or mode == 'replay':
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)
        else:
            raise Exception('Unsupported mode')

        self._set_game_mode(mode)

    def _game_init(self, with_locking=True, max_parallel=10):
        lock_file = lock = None
        if with_locking:
            lock_file = doom_lock_file(max_parallel)
            lock = FileLock(lock_file)

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                if with_locking:
                    with lock.acquire(timeout=20):
                        self.game.init()
                else:
                    self.game.init()

                break
            except Timeout:
                if with_locking:
                    log.debug(
                        'Another process currently holds the lock %s, attempt: %d', lock_file, init_attempt,
                    )
            except Exception as exc:
                log.warning('VizDoom game.init() threw an exception %r. Terminate process...', exc)
                from sample_factory.envs.env_utils import EnvCriticalError
                raise EnvCriticalError()

    def initialize(self):
        self._create_doom_game(self.mode)

        # (optional) top-down view provided by the game engine
        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = 'ffffff'
            self.game.add_game_args('+viz_am_center 1')
            self.game.add_game_args('+am_backcolor ' + background_color)
            self.game.add_game_args('+am_tswallcolor dddddd')
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args('+am_yourcolor ' + background_color)
            self.game.add_game_args('+am_cheat 0')
            self.game.add_game_args('+am_thingcolor 0000ff')  # player color
            self.game.add_game_args('+am_thingcolor_item 00ff00')
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self._game_init()
        self.initialized = True

    def _ensure_initialized(self):
        if not self.initialized:
            self.initialize()

    @staticmethod
    def _parse_variable_indices(config):
        with open(config, 'r') as config_file:
            lines = config_file.readlines()
        lines = [l.strip() for l in lines]

        variable_indices = {}

        for line in lines:
            if line.startswith('#'):
                continue  # comment

            variables_syntax = r'available_game_variables[\s]*=[\s]*\{(.*)\}'
            match = re.match(variables_syntax, line)
            if match is not None:
                variables_str = match.groups()[0]
                variables_str = variables_str.strip()
                variables = variables_str.split(' ')
                for i, variable in enumerate(variables):
                    variable_indices[variable] = i
                break

        return variable_indices

    def _black_screen(self):
        if self.black_screen is None:
            self.black_screen = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return self.black_screen

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        for variable, idx in self.variable_indices.items():
            variables[variable] = game_variables[idx]
        return variables

    def demo_path(self, episode_idx):
        demo_name = f'e{episode_idx:03d}.lmp'
        demo_path = join(self.record_to, demo_name)
        demo_path = os.path.normpath(demo_path)
        return demo_path

    def reset(self):
        self._ensure_initialized()

        if self.record_to is not None and not self.is_multiplayer:
            # does not work in multiplayer (uses different mechanism)
            if not os.path.exists(self.record_to):
                os.makedirs(self.record_to)

            demo_path = self.demo_path(self._num_episodes)
            log.warning('Recording episode demo to %s', demo_path)
            self.game.new_episode(demo_path)
        else:
            if self._num_episodes > 0:
                # no demo recording (default)
                self.game.new_episode()

        self.state = self.game.get_state()
        img = None
        try:
            img = self.state.screen_buffer
        except AttributeError:
            # sometimes Doom does not return screen buffer at all??? Rare bug
            pass

        if img is None:
            log.error('Game returned None screen buffer! This is not supposed to happen!')
            img = self._black_screen()

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self._actions_flattened = None
        self._last_episode_info = copy.deepcopy(self._prev_info)
        self._prev_info = None

        self._num_episodes += 1

        return np.transpose(img, (1, 2, 0))

    def _convert_actions(self, actions):
        """Convert actions from gym action space to the action space expected by Doom game."""

        if self.composite_action_space:
            # composite action space with multiple subspaces
            spaces = self.action_space.spaces
        else:
            # simple action space, e.g. Discrete. We still treat it like composite of length 1
            spaces = (self.action_space, )
            actions = (actions, )

        actions_flattened = []
        for i, action in enumerate(actions):
            if isinstance(spaces[i], Discretized):
                # discretized continuous action
                # check discretized first because it's a subclass of gym.spaces.Discrete
                # the order of if clauses here matters! DON'T CHANGE THE ORDER OF IFS!

                continuous_action = spaces[i].to_continuous(action)
                actions_flattened.append(continuous_action)
            elif isinstance(spaces[i], gym.spaces.Discrete):
                # standard discrete action
                num_non_idle_actions = spaces[i].n - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op

                actions_flattened.extend(action_one_hot)
            elif isinstance(spaces[i], gym.spaces.Box):
                # continuous action
                actions_flattened.extend(list(action * self.delta_actions_scaling_factor))
            else:
                raise NotImplementedError(f'Action subspace type {type(spaces[i])} is not supported!')

        return actions_flattened

    def _vizdoom_variables_bug_workaround(self, info, done):
        """Some variables don't get reset to zero on game.new_episode(). This fixes it (also check overflow?)."""
        if done and 'DAMAGECOUNT' in info:
            log.info('DAMAGECOUNT value on done: %r', info.get('DAMAGECOUNT'))

        if self._last_episode_info is not None:
            bugged_vars = ['DEATHCOUNT', 'HITCOUNT', 'DAMAGECOUNT']
            for v in bugged_vars:
                if v in info:
                    info[v] -= self._last_episode_info.get(v, 0)

    def _process_game_step(self, state, done, info):
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
            self._update_histogram(info)
            self._prev_info = copy.deepcopy(info)
        else:
            observation = self._black_screen()

            # when done=True Doom does not allow us to call get_info, so we provide info from the last frame
            info.update(self._prev_info)

        self._vizdoom_variables_bug_workaround(info, done)

        return observation, done, info

    def step(self, actions):
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        if self._actions_flattened is not None:
            # provided externally, e.g. via human play
            actions_flattened = self._actions_flattened
            self._actions_flattened = None
        else:
            actions_flattened = self._convert_actions(actions)

        default_info = {'num_frames': self.skip_frames}
        reward = self.game.make_action(actions_flattened, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        observation, done, info = self._process_game_step(state, done, default_info)
        return observation, reward, done, info

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])
            if mode == 'rgb_array':
                return img

            h, w = img.shape[:2]
            render_w = 1280

            if w < render_w:
                render_h = int(render_w * h / w)
                img = cv2.resize(img, (render_w, render_h))

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer(maxwidth=render_w)
            self.viewer.imshow(img)
            return img
        except AttributeError:
            return None

    def close(self):
        try:
            if self.game is not None:
                self.game.close()
        except RuntimeError as exc:
            log.warning('Runtime error in VizDoom game close(): %r', exc)

        if self.viewer is not None:
            self.viewer.close()

    def get_info(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())

        info_dict = {'pos': self.get_positions(variables)}
        info_dict.update(variables)
        return info_dict

    def get_info_all(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())
        info = self.get_info(variables)
        if self.previous_histogram is not None:
            info['previous_histogram'] = self.previous_histogram
        return info

    def get_positions(self, variables):
        return self._get_positions(variables)

    @staticmethod
    def _get_positions(variables):
        have_coord_data = True
        required_vars = ['POSITION_X', 'POSITION_Y', 'ANGLE']
        for required_var in required_vars:
            if required_var not in variables:
                have_coord_data = False
                break

        x = y = a = np.nan
        if have_coord_data:
            x = variables['POSITION_X']
            y = variables['POSITION_Y']
            a = variables['ANGLE']

        return {'agent_x': x, 'agent_y': y, 'agent_a': a}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info['pos']['agent_x'], info['pos']['agent_y']

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        self.current_histogram[dx, dy] += 1

    def _key_to_action(self, key):
        if hasattr(self.action_space, 'key_to_action'):
            return self.action_space.key_to_action(key)
        else:
            return key_to_action_default(key)

    def _keyboard_on_press(self, key):
        from pynput.keyboard import Key
        if key == Key.esc:
            self._terminate = True
            return False

        action = self._key_to_action(key)
        if action is not None:
            if action not in self._current_actions:
                self._current_actions.append(action)

    def _keyboard_on_release(self, key):
        action = self._key_to_action(key)
        if action is not None:
            if action in self._current_actions:
                self._current_actions.remove(action)

    # noinspection PyProtectedMember
    @staticmethod
    def play_human_mode(env, skip_frames=1, num_episodes=3, num_actions=None):
        from pynput.keyboard import Listener

        doom = env.unwrapped
        doom.skip_frames = 1  # handled by this script separately

        # noinspection PyProtectedMember
        def start_listener():
            with Listener(on_press=doom._keyboard_on_press, on_release=doom._keyboard_on_release) as listener:
                listener.join()

        listener_thread = Thread(target=start_listener)
        listener_thread.start()

        for episode in range(num_episodes):
            doom.mode = 'human'
            env.reset()
            last_render_time = time.time()
            time_between_frames = 1.0 / 35.0

            total_rew = 0.0

            while not doom.game.is_episode_finished() and not doom._terminate:
                num_actions = 14 if num_actions is None else num_actions
                turn_delta_action_idx = num_actions - 1

                actions = [0] * num_actions
                for action in doom._current_actions:
                    if isinstance(action, int):
                        actions[action] = 1  # 1 for buttons currently pressed, 0 otherwise
                    else:
                        if action == 'turn_left':
                            actions[turn_delta_action_idx] = -doom.delta_actions_scaling_factor
                        elif action == 'turn_right':
                            actions[turn_delta_action_idx] = doom.delta_actions_scaling_factor

                for frame in range(skip_frames):
                    doom._actions_flattened = actions
                    _, rew, _, _ = env.step(actions)

                    new_total_rew = total_rew + rew
                    if new_total_rew != total_rew:
                        log.info('Reward: %.3f, total: %.3f', rew, new_total_rew)
                    total_rew = new_total_rew
                    state = doom.game.get_state()

                    verbose = True
                    if state is not None and verbose:
                        info = doom.get_info()
                        print(
                            'Health:', info['HEALTH'],
                            # 'Weapon:', info['SELECTED_WEAPON'],
                            # 'ready:', info['ATTACK_READY'],
                            # 'ammo:', info['SELECTED_WEAPON_AMMO'],
                            # 'pc:', info['PLAYER_COUNT'],
                            # 'dmg:', info['DAMAGECOUNT'],
                        )

                    time_since_last_render = time.time() - last_render_time
                    time_wait = time_between_frames - time_since_last_render

                    if doom.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow('ViZDoom Automap Buffer', map_)
                        if time_wait > 0:
                            cv2.waitKey(int(time_wait) * 1000)
                    else:
                        if time_wait > 0:
                            time.sleep(time_wait)

                    last_render_time = time.time()

            if doom.show_automap:
                cv2.destroyAllWindows()

        log.debug('Press ESC to exit...')
        listener_thread.join()

    # noinspection PyProtectedMember
    @staticmethod
    def replay(env, rec_path):
        doom = env.unwrapped
        doom.mode = 'replay'
        doom._ensure_initialized()
        doom.game.replay_episode(rec_path)

        episode_reward = 0
        start = time.time()

        while not doom.game.is_episode_finished():
            doom.game.advance_action()
            r = doom.game.get_last_reward()
            episode_reward += r
            log.info('Episode reward: %.3f, time so far: %.1f s', episode_reward, time.time() - start)

        log.info('Finishing replay')
        doom.close()
