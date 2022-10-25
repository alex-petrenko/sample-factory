import gym
import numpy as np

from sample_factory.algo.utils.rl_utils import make_dones


class MultiplayerStatsWrapper(gym.Wrapper):
    """Add to info things like place in the match, gap to leader, kill-death ratio etc."""

    def __init__(self, env):
        super().__init__(env)
        self.timestep = 0
        self.prev_extra_info = dict()

    def _parse_info(self, info, done):
        if (self.timestep % 20 == 0 or done) and "FRAGCOUNT" in info:
            # no need to update these stats every frame
            kdr = info.get("FRAGCOUNT", 0.0) / (info.get("DEATHCOUNT", 0.0) + 1)
            extra_info = {"KDR": float(kdr)}

            player_count = int(info.get("PLAYER_COUNT", 1))
            player_num = int(info.get("PLAYER_NUMBER", 0))
            fragcounts = [int(info.get(f"PLAYER{pi}_FRAGCOUNT", -100000)) for pi in range(1, player_count + 1)]
            places = list(np.argsort(fragcounts))

            final_place = places.index(player_num)
            final_place = player_count - final_place  # inverse, because fragcount is sorted in increasing order
            extra_info["FINAL_PLACE"] = final_place

            if final_place > 1:
                extra_info["LEADER_GAP"] = max(fragcounts) - fragcounts[player_num]
            elif player_count > 1:
                # we won, let's log gap to 2nd place
                assert places.index(player_num) == player_count - 1
                fragcounts.sort(reverse=True)
                extra_info["LEADER_GAP"] = fragcounts[1] - fragcounts[0]  # should be negative or 0
                assert extra_info["LEADER_GAP"] <= 0
            else:
                extra_info["LEADER_GAP"] = 0

            self.prev_extra_info = extra_info
        else:
            extra_info = self.prev_extra_info

        info.update(extra_info)
        return info

    def reset(self, **kwargs):
        self.timestep = 0
        self.prev_extra_info = dict()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, reward, terminated, truncated, info

        done = make_dones(terminated, truncated)
        info = self._parse_info(info, done)
        self.timestep += 1
        return obs, reward, terminated, truncated, info
