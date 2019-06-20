import gym
import numpy as np


class DoomAdditionalInput(gym.Wrapper):
    """Add game variables to the observation space."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'measurements': gym.spaces.Box(0.0, 100.0, shape=(4, )),
        })

    @staticmethod
    def _obs_dict(obs, info):
        obs_dict = {'obs': obs, 'measurements': np.empty(4)}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))

        # same as DFP paper
        selected_weapon_ammo /= 7.5
        health = info.get('HEALTH', 0.0) / 30.0
        kills = info.get('USER2', 0.0)

        obs_dict['measurements'][0] = selected_weapon
        obs_dict['measurements'][1] = selected_weapon_ammo
        obs_dict['measurements'][2] = health
        obs_dict['measurements'][3] = kills

        return obs_dict

    def reset(self):
        obs = self.env.reset()
        info = self.env.unwrapped.get_info()
        return self._obs_dict(obs, info)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs_dict = self._obs_dict(obs, info)
        return obs_dict, rew, done, info
