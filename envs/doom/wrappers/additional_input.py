import copy
import operator
from collections import deque

import gym
import numpy as np

from algorithms.utils.algo_utils import EPS
from utils.utils import log


class DoomAdditionalInputAndRewards(gym.Wrapper):
    """Add game variables to the observation space + reward shaping."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.num_weapons = 10

        weapons_low = [0.0] * self.num_weapons
        ammo_low = [0.0] * self.num_weapons
        low = [0.0, 0.0, -1.0, -1.0, -50.0, 0.0, 0.0] + weapons_low + ammo_low

        weapons_high = [5.0] * self.num_weapons  # can have multiple weapons in the same slot?
        ammo_high = [50.0] * self.num_weapons
        high = [20.0, 50.0, 50.0, 50.0, 50.0, 1.0, 10.0] + weapons_high + ammo_high

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
            'DEATHCOUNT': (-0.75, +0.75),
            'HITCOUNT': (+0.01, -0.01),
            'DAMAGECOUNT': (+0.01, -0.01),

            'HEALTH': (+0.002, -0.002),
            'ARMOR': (+0.002, -0.001),
        }

        # without this we reward usinb BFG and shotguns too much
        self.reward_delta_limits = {'DAMAGECOUNT': 200, 'HITCOUNT': 5}

        self._prev_vars = {}

        self.weapon_preference = {
            2: 1,  # pistol
            3: 5,  # shotguns
            4: 5,  # machinegun
            5: 5,  # rocket launcher
            6: 10,  # plasmagun
            7: 10,  # bfg
        }

        for weapon in range(self.num_weapons):
            pref = self.weapon_preference.get(weapon, 1)
            self.reward_shaping_vars[f'WEAPON{weapon}'] = (+0.02 * pref, -0.01 * pref)
            self.reward_shaping_vars[f'AMMO{weapon}'] = (+0.0002 * pref, -0.0001 * pref)

        self._orig_env_reward = 0.0
        self._total_shaping_reward = 0.0
        self._episode_frames = 0

        self._selected_weapon = deque([], maxlen=5)

        self._prev_info = None

        self._reward_structure = {}

        self._verbose = False

    def _parse_info(self, obs, info, done, reset=False):
        obs_dict = {'obs': obs, 'measurements': []}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))
        self._selected_weapon.append(selected_weapon)

        # similar to DFP paper, scaling all measurements so that they are small numbers
        selected_weapon_ammo /= 15.0

        # we don't really care how much negative health we have, dead is dead
        info['HEALTH'] = max(0.0, info.get('HEALTH', 0.0))
        health = info.get('HEALTH', 0.0) / 30.0
        armor = info.get('ARMOR', 0.0) / 30.0
        kills = info.get('USER2', 0.0) / 10.0  # only works in battle and battle2, this is not really useful
        attack_ready = info.get('ATTACK_READY', 0.0)
        num_players = info.get('PLAYER_COUNT', 1) / 5.0

        measurements = obs_dict['measurements']
        measurements.append(selected_weapon)
        measurements.append(selected_weapon_ammo)
        measurements.append(health)
        measurements.append(armor)
        measurements.append(kills)
        measurements.append(attack_ready)
        measurements.append(num_players)

        for weapon in range(self.num_weapons):
            measurements.append(max(0.0, info.get(f'WEAPON{weapon}', 0.0)))
        for weapon in range(self.num_weapons):
            ammo = float(max(0.0, info.get(f'AMMO{weapon}', 0.0)))
            ammo /= 15.0  # scaling factor similar to DFP paper (to keep everything small)
            measurements.append(ammo)

        shaping_reward = 0.0
        deltas = []

        if reset:
            # skip reward calculation
            return obs_dict, shaping_reward

        if not done:
            for var_name, rewards in self.reward_shaping_vars.items():
                if var_name not in self._prev_vars:
                    continue

                # generate reward based on how the env variable values changed
                new_value = info.get(var_name, 0.0)
                prev_value = self._prev_vars[var_name]
                delta = new_value - prev_value
                if var_name in self.reward_delta_limits:
                    delta_limit = self.reward_delta_limits[var_name]
                    delta = min(delta, delta_limit)

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
                    self._reward_structure[var_name] = self._reward_structure.get(var_name, 0.0) + reward_delta

            # weapon preference reward
            weapon_reward_coeff = 0.0002
            weapon_goodness = self.weapon_preference.get(selected_weapon, 0)

            unholstered = len(self._selected_weapon) > 4 and all(sw == selected_weapon for sw in self._selected_weapon)

            if selected_weapon_ammo > 0 and unholstered:
                weapon_reward = weapon_goodness * weapon_reward_coeff
                weapon_key = f'weapon{selected_weapon}'
                deltas.append((weapon_key, weapon_reward))
                shaping_reward += weapon_reward
                self._reward_structure[weapon_key] = self._reward_structure.get(weapon_key, 0.0) + weapon_reward

        if abs(shaping_reward) > 2.5:
            log.info('Shaping reward %.3f for %r', shaping_reward, deltas)

        if done:
            sorted_rew = sorted(self._reward_structure.items(), key=operator.itemgetter(1))
            sum_rew = sum(r for key, r in sorted_rew)
            sorted_rew = {key: f'{r:.3f}' for key, r in sorted_rew}
            log.info('Sum rewards: %.3f, reward structure: %r', sum_rew, sorted_rew)

        return obs_dict, shaping_reward

    def reset(self):
        obs = self.env.reset()
        info = self.env.unwrapped.get_info()

        self._prev_vars = {}
        obs, _ = self._parse_info(obs, info, False, reset=True)

        self._prev_info = None
        self._reward_structure = {}
        self._selected_weapon.clear()

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
            info = self._prev_info
        if not done:
            # remember new variable values
            for var_name in self.reward_shaping_vars.keys():
                self._prev_vars[var_name] = info.get(var_name, 0.0)
            self._prev_info = copy.deepcopy(info)

        # log.info('Damage: %.3f hit %.3f death %.3f', info['DAMAGECOUNT'], info['HITCOUNT'], info['DEATHCOUNT'])

        return obs_dict, rew, done, info
