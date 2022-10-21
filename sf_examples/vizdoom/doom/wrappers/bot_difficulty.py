import gym

from sample_factory.utils.utils import log

# Wrapper no longer in use


class BotDifficultyWrapper(gym.Wrapper):
    """Adjust bot difficulty according to agent's final position in the match."""

    def __init__(self, env, initial_difficulty=None):
        super().__init__(env)

        self._min_difficulty = 0
        self._max_difficulty = 150
        self._difficulty_step = 10
        self._curr_difficulty = 20 if initial_difficulty is None else initial_difficulty
        self._difficulty_std = 10

        log.info("Starting with bot difficulty %d", self._curr_difficulty)

        self._adaptive_curriculum = True
        if initial_difficulty == self._max_difficulty:
            log.debug("Starting at max difficulty, disable adaptive skill curriculum")
            self._adaptive_curriculum = False

    def _analyze_standings(self, info):
        if "FINAL_PLACE" in info:
            final_place = info["FINAL_PLACE"]
            if final_place <= 1 and info.get("LEADER_GAP", 0.0) < 0:
                # we beat all the bots, increase difficulty
                self._curr_difficulty += self._difficulty_step
                self._curr_difficulty = min(self._curr_difficulty, self._max_difficulty)
            else:
                player_count = int(info.get("PLAYER_COUNT", 1))
                if final_place >= player_count - 1:
                    # got beaten badly, decrease difficulty
                    self._curr_difficulty -= self._difficulty_step
                    self._curr_difficulty = max(self._curr_difficulty, self._min_difficulty)
                else:
                    # it's fine, keep the difficulty
                    pass

    def reset(self, **kwargs):
        if hasattr(self.env.unwrapped, "bot_difficulty_mean"):
            self.env.unwrapped.bot_difficulty_mean = self._curr_difficulty
            self.env.unwrapped.bot_difficulty_std = self._difficulty_std

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, reward, terminated, truncated, info

        if (terminated | truncated) and self._adaptive_curriculum:
            self._analyze_standings(info)
        info["BOT_DIFFICULTY"] = self._curr_difficulty
        return obs, reward, terminated, truncated, info
