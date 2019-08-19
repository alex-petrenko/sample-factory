import copy

import numpy as np

from envs.doom.doom_gym import VizdoomEnv
from utils.utils import log


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(
            self,
            action_space,
            config_file,
            player_id, num_agents, max_num_players, num_bots,
            skip_frames, async_mode=False,
            record_to=None):
        super().__init__(
            action_space,
            config_file,
            skip_frames=skip_frames, async_mode=async_mode,
            record_to=record_to,
        )

        self.worker_index = 0
        self.vector_index = 0

        self.player_id = player_id
        self.num_agents = num_agents  # num agents that are not humans or bots
        self.max_num_players = max_num_players
        self.num_bots = num_bots
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

        self.is_multiplayer = True

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self, mode='algo'):
        if self.initialized:
            # Doom env already initialized!
            return

        self._create_doom_game(mode)

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

        self.game.init()

        log.info('Initialized w:%d v:%d', self.worker_index, self.vector_index)
        self.initialized = True

    def _random_bot(self, difficulty, used_bots):
        while True:
            idx = self.rng.randint(0, self.num_bots)
            bot_name = f'BOT_{difficulty}_{idx}'
            if bot_name not in used_bots:
                used_bots.append(bot_name)
                return bot_name

    def reset(self, mode='algo'):
        obs = super().reset(mode)

        if self._is_server() and self.num_bots > 0:
            self.game.send_game_command('removebots')

            bot_names = copy.deepcopy(self.bot_names)
            self.rng.shuffle(bot_names)

            used_bots = []

            for i in range(self.num_bots):
                if self.bot_difficulty_mean is None:
                    # add named bots from the list

                    if i < len(bot_names):
                        bot_name = ' ' + bot_names[i]
                    else:
                        bot_name = ''

                    # log.info('Adding bot %d %s', i, bot_name)
                    self.game.send_game_command(f'addbot{bot_name}')
                else:
                    # add random bots according to the desired difficulty
                    diff = self.rng.normal(self.bot_difficulty_mean, self.bot_difficulty_std)
                    diff = int(round(diff, -1))
                    diff = max(self.easiest_bot, diff)
                    diff = min(self.hardest_bot, diff)
                    bot_name = self._random_bot(diff, used_bots)
                    # log.info('Adding bot %d %s', i, bot_name)
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
