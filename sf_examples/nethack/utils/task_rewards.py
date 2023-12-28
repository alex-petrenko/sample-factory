import re

import numpy as np
from nle import nethack


class Score:
    def __init__(self):
        self.score = 0
        # convert name to snake_case
        # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        self.name = re.sub("(?!^)([A-Z]+)", r"_\1", self.__class__.__name__).lower()

    def reset_score(self):
        self.score = 0


class GoldScore(Score):
    def reward(self, env, last_observation, observation, end_status):
        old_blstats = last_observation[env._blstats_index]
        blstats = observation[env._blstats_index]

        old_gold = old_blstats[nethack.NLE_BL_GOLD]
        gold = blstats[nethack.NLE_BL_GOLD]

        reward = np.abs(gold - old_gold)
        self.score += reward

        return reward


class EatingScore(Score):
    def reward(self, env, last_observation, observation, end_status):
        old_internal = last_observation[env._internal_index]
        internal = observation[env._internal_index]

        reward = max(0, internal[7] - old_internal[7])
        self.score += reward

        return reward


class ScoutScore(Score):
    def __init__(self):
        super().__init__()
        self.dungeon_explored = {}

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        self.score += reward

        return reward

    def reset_score(self):
        super().reset_score()
        self.dungeon_explored = {}


class StaircaseScore(Score):
    """
    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise.
    """

    def reward(self, env, last_observation, observation, end_status):
        internal = observation[env._internal_index]
        stairs_down = internal[4]

        reward = 1 if stairs_down else 0
        self.score += reward

        return reward


class StaircasePetScore(Score):
    """
    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def reward(self, env, last_observation, observation, end_status):
        internal = observation[env._internal_index]
        stairs_down = internal[4]

        reward = 0
        if stairs_down:
            glyphs = observation[env._glyph_index]
            blstats = observation[env._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
            if np.any(nethack.glyph_is_pet(neighbors)):
                reward = 1

        self.score += reward

        return reward


class SokobanfillpitScore(Score):
    """
    This task requires the agent to put the boulders inside wholes for sokoban.
    We count each successful boulder moved into a whole as a total reward.
    """

    def reward(self, env, last_observation, observation, end_status):
        # the score counts how many pits we fill
        char_array = [chr(i) for i in observation[env._message_index]]
        message = "".join(char_array)

        if message.startswith("The boulder fills a pit.") or message.startswith(
            "The boulder falls into and plugs a whole in the floor!"
        ):
            reward = 1
        else:
            reward = 0
        self.score += reward

        return reward


class SokobansolvedlevelsScore(Score):
    def __init__(self):
        super().__init__()
        self.sokoban_levels = {}

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        # when we know that this is sokoban
        if dungeon_num == 4:
            # TODO: maybe we should count "solving" sokoban level when we reach the next level of the sokoban?
            # checking if all pits are solved can be buggy if the glyphs have different values on other levels

            # count the number of pits, glyphs SS.S_pit
            pits = np.isin(glyphs, [2411]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = pits

    def reset_score(self):
        super().reset_score()
        self.sokoban_levels = {}

    @property
    def score(self):
        score = 0
        for pits in self.sokoban_levels.values():
            # when all pits are filled we assume that sokoban level is solved
            if pits == 0:
                score += 1
        return score
