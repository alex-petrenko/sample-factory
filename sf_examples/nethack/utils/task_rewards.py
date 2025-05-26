import re

import numpy as np
from nle import nethack


class Score:
    def __init__(self):
        self.score = 0
        self.sokoban_levels = {}
        # convert name to snake_case
        # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
        self.name = re.sub("(?!^)([A-Z]+)", r"_\1", self.__class__.__name__).lower()

    def reset_score(self):
        self.score = 0


class OracleScore(Score):
    def reward(self, env, last_observation, observation, end_status):
        message = bytes(observation[env._message_index]).decode("latin-1")

        reward = 0
        # ensure that we will give reward only once
        if self.score == 0:
            if "You're in Delphi" in message or "welcome to Delphi!" in message:
                self.score = 1
                reward = 1

        return reward


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
        old_internal = last_observation[env.unwrapped._internal_index]
        internal = observation[env.unwrapped._internal_index]

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
        internal = observation[env.unwrapped._internal_index]
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
        internal = observation[env.unwrapped._internal_index]
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


class SokobanFillPitScore(Score):
    """
    This task requires the agent to put the boulders inside wholes for sokoban.
    We count each successful boulder moved into a whole as a total reward.
    """

    def reward(self, env, last_observation, observation, end_status):
        # the score counts how many pits we fill
        char_array = [chr(i) for i in observation[env._message_index]]
        message = "".join(char_array)

        if message.startswith("The boulder fills a pit.") or message.startswith(
            "The boulder falls into and plugs a hole in the floor!"
        ):
            reward = 1
        else:
            reward = 0
        self.score += reward

        return reward


class SokobanSolvedLevelsScore(Score):
    def __init__(self):
        super().__init__()
        self.sokoban_levels = {}

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        # TODO: this isn't error proof, one could create more holes / pits with a wand of digging
        # Note: we can't just check if the agent reached stairs because it can go out of the pits
        # this isn't "solving"
        if (dungeon_num, dungeon_level) == (4, 4):
            S_pit = nethack.GLYPH_CMAP_OFF + 52
            pits = np.isin(glyphs, [S_pit]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = pits
        elif (dungeon_num, dungeon_level) in [(4, 3), (4, 2), (4, 1)]:
            S_hole = nethack.GLYPH_CMAP_OFF + 54
            holes = np.isin(glyphs, [S_hole]).sum()
            key = (dungeon_num, dungeon_level)
            self.sokoban_levels[key] = holes

        self.score = 0
        for pits in self.sokoban_levels.values():
            # when all pits are filled we assume that sokoban level is solved
            if pits == 0:
                self.score += 1

    def reset_score(self):
        super().reset_score()
        self.sokoban_levels = {}


class SokobanReachedScore(Score):
    def __init__(self):
        super().__init__()
        self.reached_levels = {}

    def reward(self, env, last_observation, observation, end_status):
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        self.reached_levels[key] = 1

        self.score = 0
        for reached in self.reached_levels.keys():
            if reached in [(4, 4), (4, 3), (4, 2), (4, 1)]:
                self.score += 1

    def reset_score(self):
        super().reset_score()
        self.reached_levels = {}


class SokobanSolvedScore(Score):
    def __init__(self):
        super().__init__()

    def reward(self, env, last_observation, observation, end_status):
        glyphs = observation[env._glyph_index]
        blstats = observation[env._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        S_vcdoor = nethack.GLYPH_CMAP_OFF + 15  # /* closed door, vertical wall */
        S_hcdoor = nethack.GLYPH_CMAP_OFF + 16  # /* closed door, horizontal wall */
        door_closed = frozenset({S_vcdoor, S_hcdoor})
        if (dungeon_num, dungeon_level) == (4, 1) and np.isin(glyphs, [door_closed]).sum() == 0:
            self.score = 1

    def reset_score(self):
        super().reset_score()
