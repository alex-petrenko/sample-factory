from collections import deque

import gym
import numpy as np

from algorithms.utils.algo_utils import EPS
from utils.utils import log


class DoomAdditionalInputAndRewards(gym.Wrapper):
    """Add game variables to the observation space."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.num_weapons = 10

        weapons_low = [0.0] * self.num_weapons
        ammo_low = [0.0] * self.num_weapons
        low = [0.0, 0.0, -1.0, -1.0, -50.0, 0.0] + weapons_low + ammo_low

        weapons_high = [5.0] * self.num_weapons  # can have multiple weapons in the same slot?
        ammo_high = [50.0] * self.num_weapons
        high = [20.0, 50.0, 50.0, 50.0, 50.0, 1.0] + weapons_high + ammo_high

        self.observation_space = gym.spaces.Dict({
            'obs': current_obs_space,
            'measurements': gym.spaces.Box(
                low=np.array(low), high=np.array(high),
            ),
        })

        # game variables to use for reward shaping
        # plus corresponding reward values for positive and negative delta (per unit)
        self.reward_shaping_vars = {
            'FRAGCOUNT': (+1, -1.5),
            'DEATHCOUNT': (-0.25, +0.25),
            'HITCOUNT': (+0.01, -0.01),
            'DAMAGECOUNT': (+0.01, -0.01),

            'HEALTH': (+0.002, -0.002),
            'ARMOR': (+0.002, -0.001),
        }

        for weapon in range(self.num_weapons):
            self.reward_shaping_vars[f'WEAPON{weapon}'] = (+0.2, -0.1)
            self.reward_shaping_vars[f'AMMO{weapon}'] = (+0.002, -0.00001)

        self.prev_vars = {}

        self.weapon_preference = {
            2: 1,  # pistol
            3: 5,  # shotguns
            4: 10,  # machinegun
            5: 5,  # rocket launcher
            6: 10,  # plasmagun
            7: 20,  # bfg
        }

        self._orig_env_reward = 0.0
        self._total_shaping_reward = 0.0
        self._episode_frames = 0

        self._selected_weapon = deque([], maxlen=5)

        self._prev_info = None

        self._verbose = False

    def _parse_info(self, obs, info, done):
        obs_dict = {'obs': obs, 'measurements': []}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))
        self._selected_weapon.append(selected_weapon)

        # similar to DFP paper
        selected_weapon_ammo /= 15.0

        # we don't really care how much negative health we have, dead is dead
        info['HEALTH'] = max(0.0, info.get('HEALTH', 0.0))
        health = info.get('HEALTH', 0.0) / 30.0
        armor = info.get('ARMOR', 0.0) / 30.0
        kills = info.get('USER2', 0.0) / 10.0  # only works in battle and battle2, this is not really useful
        attack_ready = info.get('ATTACK_READY', 0.0)

        obs_dict['measurements'].append(selected_weapon)
        obs_dict['measurements'].append(selected_weapon_ammo)
        obs_dict['measurements'].append(health)
        obs_dict['measurements'].append(armor)
        obs_dict['measurements'].append(kills)
        obs_dict['measurements'].append(attack_ready)

        for weapon in range(self.num_weapons):
            obs_dict['measurements'].append(max(0.0, info.get(f'WEAPON{weapon}', 0.0)))
        for weapon in range(self.num_weapons):
            ammo = float(max(0.0, info.get(f'AMMO{weapon}', 0.0)))
            ammo /= 15.0  # scaling factor similar to DFP paper (to keep everything small)
            obs_dict['measurements'].append(ammo)

        shaping_reward = 0.0
        deltas = []

        if not done:
            for var_name, rewards in self.reward_shaping_vars.items():
                if var_name not in self.prev_vars:
                    continue

                # generate reward based on how the env variable values changed
                new_value = info.get(var_name, 0.0)
                prev_value = self.prev_vars[var_name]
                delta = new_value - prev_value
                reward_delta = 0
                if delta > EPS:
                    reward_delta = delta * rewards[0]
                elif delta < -EPS:
                    reward_delta = -delta * rewards[1]

                # player_id = 1
                # if hasattr(self.env.unwrapped, 'player_id'):
                #     player_id = self.env.unwrapped.player_id

                shaping_reward += reward_delta

                if abs(reward_delta) > EPS:
                    deltas.append((var_name, reward_delta, delta))

            # weapon preference reward
            weapon_reward_coeff = 0.0002
            weapon_goodness = self.weapon_preference.get(selected_weapon, 0)

            unholstered = len(self._selected_weapon) > 4 and all(sw == selected_weapon for sw in self._selected_weapon)

            if selected_weapon_ammo > 0 and unholstered:
                weapon_reward = weapon_goodness * weapon_reward_coeff
                deltas.append((f'weapon{selected_weapon}', weapon_reward))
                shaping_reward += weapon_reward

        if abs(shaping_reward) > 0.5:
            log.info('Shaping reward %.3f for %r', shaping_reward, deltas)

        # remember new variable values
        for var_name in self.reward_shaping_vars.keys():
            self.prev_vars[var_name] = info.get(var_name, 0.0)

        return obs_dict, shaping_reward

    def reset(self):
        self.prev_vars = {}
        self._prev_info = None
        self._selected_weapon.clear()

        obs = self.env.reset()
        info = self.env.unwrapped.get_info()
        obs, _ = self._parse_info(obs, info, False)

        self._orig_env_reward = self._total_shaping_reward = 0.0
        self._episode_frames = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            return obs, rew, done, info

        self._orig_env_reward += rew

        obs_dict, shaping_rew = self._parse_info(obs, info, done)
        rew += shaping_rew
        self._total_shaping_reward += shaping_rew
        self._episode_frames += 1

        if self._verbose:
            log.info('Original env reward before shaping: %.3f', self._orig_env_reward)
            player_id = 1
            if hasattr(self.env.unwrapped, 'player_id'):
                player_id = self.env.unwrapped.player_id

            log.info(
                'Total shaping reward is %.3f for %d (done %d)',
                self._total_shaping_reward, player_id, done,
            )

        if done and self._prev_info is not None:
            info.update(self._prev_info)
        self._prev_info = info

        # if abs(rew) > 0.5:
        #     log.info(
        #         'Final rew: %.4f, total env_rew %.4f, total shaping rew %.4f, shaping %.4f',
        #         rew, self._orig_env_reward, self._total_shaping_reward, shaping_rew,
        #     )

        return obs_dict, rew, done, info
