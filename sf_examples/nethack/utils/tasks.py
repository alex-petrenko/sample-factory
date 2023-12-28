import enum

import numpy as np
from nle import nethack
from nle.env.tasks import NetHackChallenge


class NetHackGold(NetHackChallenge):
    """Environment for the "gold" task.
    The task is similar to the one defined by `NetHackScore`, but the reward
    uses changes in the amount of gold collected by the agent, rather than the
    score.
    The agent will pickup gold automatically by walking on top of it.
    """

    def __init__(self, *args, **kwargs):
        options = kwargs.pop("options", None)

        if options is None:
            # Copy & swap out "pickup_types".
            options = []
            for option in nethack.NETHACKOPTIONS:
                if option.startswith("pickup_types"):
                    options.append("pickup_types:$")
                    continue
                options.append(option)

        super().__init__(*args, options=options, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Difference between previous gold and new gold."""
        del end_status  # Unused
        del action  # Unused
        if not self.nethack.in_normal_game():
            # Before game started and after it ended blstats are zero.
            return 0.0

        old_blstats = last_observation[self._blstats_index]
        blstats = observation[self._blstats_index]

        old_gold = old_blstats[nethack.NLE_BL_GOLD]
        gold = blstats[nethack.NLE_BL_GOLD]

        time_penalty = self._get_time_penalty(last_observation, observation)

        return gold - old_gold + time_penalty


class NetHackStaircase(NetHackChallenge):
    """Environment for "staircase" task.
    This task requires the agent to get on top of a staircase down (>).
    The reward function is :math:`I + \text{TP}`, where :math:`I` is 1 if the
    task is successful, and 0 otherwise, and :math:`\text{TP}` is the time step
    function as defined by `NetHackScore`.
    """

    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1
        TASK_SUCCESSFUL = 2

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action  # Unused
        time_penalty = self._get_time_penalty(last_observation, observation)
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = 1
        else:
            reward = 0
        return reward + time_penalty


class NetHackStaircasePet(NetHackStaircase):
    """Environment for "staircase-pet" task.
    This task requires the agent to get on top of a staircase down (>), while
    having their pet next to it. See `NetHackStaircase` for the reward function.
    """

    def _is_episode_end(self, observation):
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            glyphs = observation[self._glyph_index]
            blstats = observation[self._blstats_index]
            x, y = blstats[:2]

            neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
            if np.any(nethack.glyph_is_pet(neighbors)):
                return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


class NetHackOracle(NetHackStaircase):
    """Environment for "oracle" task.
    This task requires the agent to reach the oracle (by standing next to it).
    See `NetHackStaircase` for the reward function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_glyph = None
        for glyph in range(nethack.GLYPH_MON_OFF, nethack.GLYPH_PET_OFF):
            if nethack.permonst(nethack.glyph_to_mon(glyph)).mname == "Oracle":
                self.oracle_glyph = glyph
                break
        assert self.oracle_glyph is not None

    def _is_episode_end(self, observation):
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]
        x, y = blstats[:2]

        neighbors = glyphs[y - 1 : y + 2, x - 1 : x + 2]
        if np.any(neighbors == self.oracle_glyph):
            return self.StepStatus.TASK_SUCCESSFUL
        return self.StepStatus.RUNNING


# FIXME: the way the reward function is currently structured means the
# agents gets a penalty of -1 every other step (since the
# uhunger increases by that)
# thus the step penalty becomes irrelevant
class NetHackEat(NetHackChallenge):
    """Environment for the "eat" task.
    The task is similar to the one defined by `NetHackScore`, but the reward
    uses positive changes in the character's hunger level (e.g. by consuming
    comestibles or monster corpses), rather than the score.
    """

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Difference between previous hunger and new hunger."""
        del end_status  # Unused
        del action  # Unused

        if not self.nethack.in_normal_game():
            # Before game started and after it ended blstats are zero.
            return 0.0

        old_internal = last_observation[self._internal_index]
        internal = observation[self._internal_index]

        old_uhunger = old_internal[7]
        uhunger = internal[7]

        reward = max(0, uhunger - old_uhunger)

        time_penalty = self._get_time_penalty(last_observation, observation)

        return reward + time_penalty


class NetHackScout(NetHackChallenge):
    """Environment for the "scout" task.
    The task is similar to the one defined by `NetHackScore`, but the score is
    defined by the changes in glyphs discovered by the agent.
    """

    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        del end_status  # Unused
        del action  # Unused

        if not self.nethack.in_normal_game():
            # Before game started and after it ended blstats are zero.
            return 0.0

        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0
        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        time_penalty = self._get_time_penalty(last_observation, observation)
        return reward + time_penalty
