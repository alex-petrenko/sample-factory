import gym
import numpy as np

from algorithms.utils.algo_utils import EPS
from utils.utils import log


class DoomAdditionalInputAndRewards(gym.Wrapper):
    """Add game variables to the observation space."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'measurements': gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1e3]), high=np.array([20.0, 1000.0, 100.0, 1e3]),
            ),
        })

        # game variables to use for reward shaping
        # plus corresponding reward values for positive and negative delta (per unit)
        self.reward_shaping_vars = {
            'FRAGCOUNT': (+1, -1),
            'DEATHCOUNT': (-1, +1),
            'HITCOUNT': (+0.5, -0.5),

            'HEALTH': (+0.002, -0.002),
            'SELECTED_WEAPON_AMMO': (+0.01, 0.0),
        }

        self.prev_vars = {}
        self._reset_vars()

    def _reset_vars(self):
        for k in self.reward_shaping_vars.keys():
            self.prev_vars[k] = 0.0

        # specific values
        self.prev_vars['HEALTH'] = 100.0
        self.prev_vars['SELECTED_WEAPON_AMMO'] = 100.0

    def _parse_info(self, obs, info):
        obs_dict = {'obs': obs, 'measurements': np.empty(4)}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))

        # same as DFP paper
        selected_weapon_ammo /= 7.5

        # we don't really care how much negative health we have, dead is dead
        info['HEALTH'] = max(0.0, info.get('HEALTH', 0.0))
        health = info.get('HEALTH', 0.0) / 30.0
        kills = info.get('USER2', 0.0) / 10.0  # only works in battle and battle2, this is not really useful

        obs_dict['measurements'][0] = selected_weapon
        obs_dict['measurements'][1] = selected_weapon_ammo
        obs_dict['measurements'][2] = health
        obs_dict['measurements'][3] = kills

        shaping_reward = 0.0

        for var_name, rewards in self.reward_shaping_vars.items():
            new_value = info.get(var_name, 0.0)
            prev_value = self.prev_vars[var_name]
            delta = new_value - prev_value
            reward_delta = 0
            if delta > EPS:
                reward_delta = delta * rewards[0]
            elif delta < -EPS:
                reward_delta = -delta * rewards[1]

            # if abs(reward_delta) > EPS and self.env.unwrapped.player_id == 1:
            #     log.info('Reward %.3f for %s, delta %.3f (player %r)', reward_delta, var_name, delta, self.env.unwrapped.player_id)

            shaping_reward += reward_delta

        # remember new variable values
        for var_name in self.reward_shaping_vars.keys():
            self.prev_vars[var_name] = info.get(var_name, 0.0)

        # if print_info and self.env.unwrapped.player_id == 1:
        #     log.info('DEATHCOUNT: %d, FRAGCOUNT %d, KILLCOUNT %d, DEAD %d (player %r)', info.get('DEATHCOUNT', -42), info.get('FRAGCOUNT', -42), info.get('KILLCOUNT', -42), info.get('DEAD', -42), self.env.unwrapped.player_id)

        return obs_dict, shaping_reward

    def reset(self):
        self._reset_vars()

        obs = self.env.reset()
        info = self.env.unwrapped.get_info()
        obs, _ = self._parse_info(obs, info)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            return obs, rew, done, info

        obs_dict, shaping_rew = self._parse_info(obs, info)
        rew += shaping_rew

        # if abs(rew) > EPS and self.env.unwrapped.player_id == 1:
        #     log.info('Total reward for the agent %r is %.3f', self.env.unwrapped.player_id, rew)

        return obs_dict, rew, done, info
