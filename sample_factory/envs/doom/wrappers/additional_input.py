import gym
import numpy as np

from sample_factory.envs.doom.wrappers.reward_shaping import NUM_WEAPONS


class DoomAdditionalInput(gym.Wrapper):
    """Add game variables to the observation space + reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.num_weapons = NUM_WEAPONS

        weapons_low = [0.0] * self.num_weapons
        ammo_low = [0.0] * self.num_weapons
        low = [0.0, 0.0, -1.0, -1.0, -50.0, 0.0, 0.0] + weapons_low + ammo_low

        weapons_high = [5.0] * self.num_weapons  # can have multiple weapons in the same slot?
        ammo_high = [50.0] * self.num_weapons
        high = [20.0, 50.0, 50.0, 50.0, 50.0, 1.0, 10.0] + weapons_high + ammo_high

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'measurements': gym.spaces.Box(
                low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32),
            ),
        })

        num_measurements = len(low)
        self.measurements_vec = np.zeros([num_measurements], dtype=np.float32)

    def _parse_info(self, obs, info):
        obs_dict = {'obs': obs, 'measurements': self.measurements_vec}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0))

        # similar to DFP paper, scaling all measurements so that they are small numbers
        selected_weapon_ammo /= 15.0
        selected_weapon_ammo = min(selected_weapon_ammo, 5.0)

        # we don't really care how much negative health we have, dead is dead
        info['HEALTH'] = max(0.0, info.get('HEALTH', 0.0))
        health = info.get('HEALTH', 0.0) / 30.0
        armor = info.get('ARMOR', 0.0) / 30.0
        kills = info.get('USER2', 0.0) / 10.0  # only works in battle and battle2, this is not really useful
        attack_ready = info.get('ATTACK_READY', 0.0)
        num_players = info.get('PLAYER_COUNT', 1) / 5.0

        # TODO add FRAGCOUNT to the input, so agents know when they are winning/losing

        measurements = obs_dict['measurements']

        i = 0
        measurements[i] = float(selected_weapon)
        i += 1
        measurements[i] = float(selected_weapon_ammo)
        i += 1
        measurements[i] = float(health)
        i += 1
        measurements[i] = float(armor)
        i += 1
        measurements[i] = float(kills)
        i += 1
        measurements[i] = float(attack_ready)
        i += 1
        measurements[i] = float(num_players)
        i += 1

        for weapon in range(self.num_weapons):
            measurements[i] = float(max(0.0, info.get(f'WEAPON{weapon}', 0.0)))
            i += 1
        for weapon in range(self.num_weapons):
            ammo = float(max(0.0, info.get(f'AMMO{weapon}', 0.0)))
            ammo /= 15.0  # scaling factor similar to DFP paper (to keep everything small)
            ammo = min(ammo, 5.0)  # to avoid values that are too big
            measurements[i] = ammo
            i += 1

        return obs_dict

    def reset(self):
        obs = self.env.reset()
        info = self.env.unwrapped.get_info()
        obs = self._parse_info(obs, info)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            return obs, rew, done, info

        obs_dict = self._parse_info(obs, info)
        return obs_dict, rew, done, info
