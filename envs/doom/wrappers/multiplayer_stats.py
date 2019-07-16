import gym
import numpy as np


class MultiplayerStatsWrapper(gym.Wrapper):
    """Add to info things like place in the match, gap to leader, kill-death ratio etc."""

    def __init__(self, env):
        super().__init__(env)
        self._timestep = 0
        self._prev_extra_info = {}

    def _parse_info(self, info):
        if self._timestep % 10 == 0 and 'FRAGCOUNT' in info:
            # no need to update these stats every frame
            kdr = info.get('FRAGCOUNT', 0.0) / (info.get('DEATHCOUNT', 0.0) + 1)
            extra_info = {'KDR': kdr}

            player_count = int(info.get('PLAYER_COUNT', 1))
            player_num = int(info.get('PLAYER_NUM', 1))
            fragcounts = [int(info.get(f'PLAYER{pi}_FRAGCOUNT', -100000)) for pi in range(1, player_count + 1)]
            places = list(np.argsort(fragcounts))

            final_place = places[player_num - 1]
            final_place = player_count - final_place  # inverse, because fragcount is sorted in increasing order
            extra_info['FINAL_PLACE'] = final_place

            if final_place > 1:
                extra_info['LEADER_GAP'] = max(fragcounts) - fragcounts[player_num - 1]
            elif player_num > 1:
                # we won, let's log gap to 2nd place
                assert places[player_num - 1] == player_num - 1
                fragcounts.sort(reverse=True)
                extra_info['LEADER_GAP'] = fragcounts[0] - fragcounts[1]  # should be negative or 0
                assert extra_info['LEADER_GAP'] <= 0

            self._prev_extra_info = extra_info
        else:
            extra_info = self._prev_extra_info

        info.update(extra_info)
        return info

    def reset(self, **kwargs):
        self._timestep = 0
        self._prev_extra_info = {}
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = self._parse_info(info)
        self._timestep += 1
        return obs, reward, done, info
