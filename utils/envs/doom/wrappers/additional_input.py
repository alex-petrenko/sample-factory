import gym
import numpy as np

from utils.utils import log


class DoomAdditionalInput(gym.Wrapper):
    """Add game variables to the observation space."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'measurements': gym.spaces.Box(0.0, 100.0, shape=(4, )),
        })

        self.prev_health = 100.0
        self.prev_ammo = 100.0
        self.first_frame = True  # ammo is delivered only after 1 frame

    def _obs_dict(self, obs, info):
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

        shaping_reward = 0.0
        if not self.first_frame:
            if health > self.prev_health:
                shaping_reward += 0.5
                # log.info('Health reward!')
            if selected_weapon_ammo > self.prev_ammo:
                shaping_reward += 0.5
                # log.info('Ammo reward! %r %r', selected_weapon_ammo, self.prev_ammo)

        self.prev_health = health
        self.prev_ammo = selected_weapon_ammo

        return obs_dict, shaping_reward

    def reset(self):
        self.prev_health = 100.0
        self.prev_ammo = 100.0
        self.first_frame = True

        obs = self.env.reset()
        info = self.env.unwrapped.get_info()
        obs, _ = self._obs_dict(obs, info)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            return obs, rew, done, info

        obs_dict, shaping_rew = self._obs_dict(obs, info)
        self.first_frame = False
        return obs_dict, rew + shaping_rew, done, info
