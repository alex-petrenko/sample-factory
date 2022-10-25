import copy
import os
from typing import Optional

from sample_factory.utils.network import is_udp_port_available
from sample_factory.utils.utils import log
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

DEFAULT_UDP_PORT = int(os.environ.get("DOOM_DEFAULT_UDP_PORT", 40300))
# log.info('Default UDP port is %r', DEFAULT_UDP_PORT)

# This try except block is to increase the env timeout connection flag in travis
try:
    vizdoom_env_timeout = int(os.environ["TRAVIS_VIZDOOM_ENV_TIMEOUT"])
except KeyError:
    vizdoom_env_timeout = 4


def find_available_port(start_port, increment=1000):
    port = start_port
    while port < 65535 and not is_udp_port_available(port):
        port += increment

    log.debug("Port %r is available", port)
    return port


class VizdoomEnvMultiplayer(VizdoomEnv):
    def __init__(
        self,
        action_space,
        config_file,
        player_id,
        num_agents,
        max_num_players,
        num_bots,
        skip_frames,
        async_mode=False,
        respawn_delay=0,
        timelimit=0.0,
        record_to=None,
        render_mode: Optional[str] = None,
    ):
        super().__init__(
            action_space,
            config_file,
            skip_frames=skip_frames,
            async_mode=async_mode,
            record_to=record_to,
            render_mode=render_mode,
        )

        self.worker_index = 0
        self.vector_index = 0

        self.player_id = player_id
        self.num_agents = num_agents  # num agents that are not humans or bots
        self.max_num_players = max_num_players
        self.num_bots = num_bots
        self.timestep = 0
        self.update_state = True

        # # Removed bot curriculum learning in favor of randomly generated bots
        # # hardcode bot names for consistency if needed
        # self.bot_names = [
        #     "Blazkowicz",
        #     "PerfectBlue",
        #     "PerfectRed",
        #     "PerfectGreen",
        #     "PerfectPurple",
        #     "PerfectYellow",
        #     "PerfectWhite",
        #     "PerfectLtGreen",
        # ]
        # self.bot_difficulty_mean = self.bot_difficulty_std = None
        # self.hardest_bot = 100
        # self.easiest_bot = 10

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        self.is_multiplayer = True
        self.init_info = None

    def _is_server(self):
        return self.player_id == 0

    def _ensure_initialized(self):
        if self.initialized:
            # Doom env already initialized!
            return

        self._create_doom_game(self.mode)
        port = DEFAULT_UDP_PORT if self.init_info is None else self.init_info.get("port", DEFAULT_UDP_PORT)

        if self._is_server():
            log.info("Using port %d on host...", port)
            if not is_udp_port_available(port):
                raise Exception("Port %r unavailable", port)

            # This process will function as a host for a multiplayer game with this many players (including the host).
            # It will wait for other machines to connect using the -join parameter and then
            # start the game when everyone is connected.
            game_args_list = [
                f"-host {self.max_num_players}",
                f"-port {port}",
                "-deathmatch",  # Deathmatch rules are used for the game.
                f"+timelimit {self.timelimit}",  # The game (episode) will end after this many minutes have elapsed.
                "+sv_forcerespawn 1",  # Players will respawn automatically after they die.
                "+sv_noautoaim 1",  # Autoaim is disabled for all players.
                "+sv_respawnprotect 1",  # Players will be invulnerable for two second after spawning.
                "+sv_spawnfarthest 1",  # Players will be spawned as far as possible from any other players.
                "+sv_nocrouch 1",  # Disables crouching.
                "+sv_nojump 1",  # Disables jumping.
                "+sv_nofreelook 1",  # Disables free look with a mouse (only keyboard).
                "+sv_noexit 1",  # Prevents players from exiting the level in deathmatch before timelimit is hit.
                f"+viz_respawn_delay {self.respawn_delay}",  # Sets delay between respanws (in seconds).
                f"+viz_connect_timeout {vizdoom_env_timeout}",
            ]
            self.game.add_game_args(" ".join(game_args_list))

            # Additional commands:
            #
            # disables depth and labels buffer and the ability to use commands
            # that could interfere with multiplayer game (should use this in evaluation)
            # '+viz_nocheat 1'

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f"+name AI{self.player_id}_host +colorset 0")

            if self.record_to is not None:
                # reportedly this does not work with bots
                demo_path = self.demo_path(self._num_episodes, self.record_to)
                log.debug("Recording multiplayer demo to %s", demo_path)
                self.game.add_game_args(f"-record {demo_path}")
        else:
            # Join existing game.
            self.game.add_game_args(
                f"-join 127.0.0.1:{port} "  # Connect to a host for a multiplayer game.
                f"+viz_connect_timeout {vizdoom_env_timeout} "
            )

            # Name your agent and select color
            # colors:
            # 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            self.game.add_game_args(f"+name AI{self.player_id} +colorset 0")

        self.game.set_episode_timeout(int(self.timelimit * 60 * self.game.get_ticrate()))

        self._game_init(with_locking=False)  # locking is handled by the multi-agent wrapper
        log.info("Initialized w:%d v:%d player:%d", self.worker_index, self.vector_index, self.player_id)
        self.initialized = True

    # def _random_bot(self, difficulty, used_bots):
    #     while True:
    #         idx = self.rng.integers(0, self.num_bots)
    #         bot_name = f"BOT_{difficulty}_{idx}"
    #         if bot_name not in used_bots:
    #             used_bots.append(bot_name)
    #             return bot_name

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        if self._is_server() and self.num_bots > 0:
            self.game.send_game_command("removebots")

            for _ in range(self.num_bots):
                self.game.send_game_command("addbot")

            # # No longer use curriculum learning
            # bot_names = copy.deepcopy(self.bot_names)
            # self.rng.shuffle(bot_names)
            # used_bots = []
            # for i in range(self.num_bots):
            #     if self.bot_difficulty_mean is None:
            #         # add named bots from the list
            #         if i < len(bot_names):
            #             bot_name = " " + bot_names[i]
            #         else:
            #             bot_name = ""
            #         # log.info('Adding bot %d %s', i, bot_name)
            #         self.game.send_game_command(f"addbot{bot_name}")
            #     else:
            #         # add random bots according to the desired difficulty
            #         diff = self.rng.normal(self.bot_difficulty_mean, self.bot_difficulty_std)
            #         diff = int(round(diff, -1))
            #         diff = max(self.easiest_bot, diff)
            #         diff = min(self.hardest_bot, diff)
            #         bot_name = self._random_bot(diff, used_bots)
            #         # log.info('Adding bot %d %s', i, bot_name)
            #         self.game.send_game_command(f"addbot {bot_name}")

        self.timestep = 0
        self.update_state = True
        return obs, info

    def step(self, actions):
        if self.skip_frames > 1 or self.num_agents == 1:
            # not used in multi-agent mode due to VizDoom limitations
            # this means that we have only one agent (+ maybe some bots, which is why we're in multiplayer mode)
            return super().step(actions)

        self._ensure_initialized()

        actions_binary = self._convert_actions(actions)

        self.game.set_action(actions_binary)
        self.game.advance_action(1, self.update_state)
        self.timestep += 1

        if not self.update_state:
            return None, None, None, None, None

        state = self.game.get_state()
        reward = self.game.get_last_reward()
        terminated = self.game.is_episode_finished()

        if self.record_to is not None:
            # send 'stop recording' command 1 tick before the end of the episode
            # otherwise it does not get saved to disk
            if self.game.get_episode_time() + 1 == self.game.get_episode_timeout():
                log.debug("Calling stop recording command!")
                self.game.send_game_command("stop")

        observation, terminated, info = self._process_game_step(state, terminated, {})
        truncated = False
        return observation, reward, terminated, truncated, info
