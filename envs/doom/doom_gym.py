import os
import re
import time
from os.path import join
from threading import Thread

import cv2
import gym
import numpy as np
from gym.utils import seeding
from pynput.keyboard import Listener, Key

from vizdoom.vizdoom import ScreenResolution, DoomGame, Mode, AutomapMode

from envs.doom.doom_helpers import key_to_action
from utils.utils import log


class VizdoomEnv(gym.Env):

    def __init__(self,
                 action_space,
                 config_file,
                 coord_limits=None,
                 max_histogram_length=200,
                 show_automap=False,
                 skip_frames=1):
        self.initialized = False

        # essential game data
        self.game = None
        self.state = None
        self.curr_seed = 0
        self.rng = None
        self.skip_frames = skip_frames

        # optional - for topdown view rendering and visitation heatmaps
        self.show_automap = show_automap
        self.coord_limits = coord_limits

        # can be adjusted after the environment is created (but before any reset() call) via observation space wrapper
        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        # provided as a part of environment definition, since these depend on the scenario and
        # can be quite complex multi-discrete spaces
        self.action_space = action_space

        # treat 0th action as the actual action instead of no-op
        self.no_idle_action = False

        scenarios_dir = join(os.path.dirname(__file__), 'scenarios')
        self.config_path = join(scenarios_dir, config_file)
        self.variable_indices = self._parse_variable_indices(self.config_path)

        # only created if we call render() method
        self.viewer = None

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

        self.seed()

    def seed(self, seed=None):
        self.curr_seed = seeding.hash_seed(seed, max_bytes=4)
        self.rng, _ = seeding.np_random(seed=self.curr_seed)
        return [self.curr_seed, self.rng]

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self.game = DoomGame()

        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        self.game.set_seed(self.rng.randint(0, 2**32 - 1))

        if mode == 'algo':
            self.game.set_window_visible(False)
        elif mode == 'human':
            self.game.add_game_args('+freelook 1')
            self.game.set_window_visible(True)

            # another option is to use spectator mode, then the game handles keyboard for you
            self.game.set_mode(Mode.PLAYER)
        else:
            raise Exception('Unsupported mode')

        # (optional) top-down view provided by the game engine
        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = 'ffffff'
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self.game.init()

        self.initialized = True

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

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        for variable, idx in self.variable_indices.items():
            variables[variable] = game_variables[idx]
        return variables

    def reset(self, mode='algo'):
        self._ensure_initialized(mode)

        self.game.new_episode()
        self.state = self.game.get_state()
        img = self.state.screen_buffer

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        return np.transpose(img, (1, 2, 0))

    def _convert_actions(self, actions):
        """
        Convert actions from gym action space to the action space expected by Doom game.
        TODO: continuous actions for aiming?
        """
        if not isinstance(actions, (tuple, list)):
            actions = (actions,)

        spaces = self.action_space.spaces if hasattr(self.action_space, 'spaces') else (self.action_space, )

        actions_binary = []
        for i, action in enumerate(actions):
            num_non_idle_actions = spaces[i].n if self.no_idle_action else spaces[i].n - 1
            action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)

            if self.no_idle_action:
                action_one_hot[action] = 1
            elif action > 0:
                action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op

            actions_binary.extend(action_one_hot)

        return actions_binary

    def step(self, actions):
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        self._ensure_initialized()
        info = {'num_frames': self.skip_frames}

        actions_binary = self._convert_actions(actions)

        reward = self.game.make_action(actions_binary, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
            self._update_histogram(info)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)

        return observation, reward, done, info

    def render(self, mode='human'):
        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])

            h, w = img.shape[:2]
            render_w = 1280

            if w < render_w:
                render_h = int(render_w * h / w)
                img = cv2.resize(img, (render_w, render_h))

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer(maxwidth=render_w)
            self.viewer.imshow(img)
        except AttributeError:
            pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _keyboard_on_press(self, key):
        if key == Key.esc:
            self._terminate = True
            return False

        action = key_to_action(key)
        if action is not None:
            if action not in self._current_actions:
                self._current_actions.append(action)

    def _keyboard_on_release(self, key):
        action = key_to_action(key)
        if action is not None:
            if action in self._current_actions:
                self._current_actions.remove(action)

    def play_human_mode(self, skip_frames=1, num_episodes=3):
        def start_listener():
            with Listener(on_press=self._keyboard_on_press, on_release=self._keyboard_on_release) as listener:
                listener.join()

        listener_thread = Thread(target=start_listener)
        listener_thread.start()

        for episode in range(num_episodes):
            self.reset('human')
            last_render_time = time.time()
            time_between_frames = 1.0 / 35.0

            while not self.game.is_episode_finished() and not self._terminate:
                num_actions = 8  # hardcoded here for simplicity
                actions = [0] * num_actions
                for action in self._current_actions:
                    actions[action] = 1  # 1 for buttons currently pressed, 0 otherwise

                for frame in range(skip_frames):
                    self.game.make_action(actions, 1)
                    log.info('Action taken: %r', actions)
                    state = self.game.get_state()
                    total_reward = self.game.get_total_reward()

                    verbose = False
                    if state is not None and verbose:
                        print('===============================')
                        print('Info: ', self.get_info())
                        print('State: #' + str(state.number))
                        print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
                        print('Reward: \t' + str(self.game.get_last_reward()))
                        print('Total Reward: \t' + str(total_reward))

                    time_since_last_render = time.time() - last_render_time
                    time_wait = time_between_frames - time_since_last_render

                    if self.show_automap and state.automap_buffer is not None:
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

            if self.show_automap:
                cv2.destroyAllWindows()

        log.debug('Press ESC to exit...')
        listener_thread.join()

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
