import gym
import numpy as np

from algorithms.utils.algo_utils import EPS


class DoomAdditionalInputAndRewards(gym.Wrapper):
    """Add game variables to the observation space."""

    def __init__(self, env):
        super().__init__(env)
        current_obs_space = self.observation_space

        self.num_weapons = 10

        weapons_low = [0.0] * self.num_weapons
        ammo_low = [0.0] * self.num_weapons
        low = [0.0, 0.0, -1.0, -1.0, -1e3] + weapons_low + ammo_low

        weapons_high = [5.0] * self.num_weapons  # can have multiple weapons in the same slot?
        ammo_high = [1000.0] * self.num_weapons
        high = [20.0, 1000.0, 100.0, 100.0, 1e3] + weapons_high + ammo_high

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
            'HITCOUNT': (+0.05, -0.05),

            'HEALTH': (+0.002, -0.002),
            'ARMOR': (+0.002, -0.001),
        }

        for weapon in range(self.num_weapons):
            self.reward_shaping_vars[f'WEAPON{weapon}'] = (+0.2, -0.1)
            self.reward_shaping_vars[f'AMMO{weapon}'] = (+0.002, -0.001)

        self.prev_vars = {}

        self._orig_env_reward = 0.0
        self._total_shaping_reward = 0.0
        self._episode_frames = 0

    def _parse_info(self, obs, info, done):
        obs_dict = {'obs': obs, 'measurements': []}

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = round(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))

        # same as DFP paper
        selected_weapon_ammo /= 7.5

        # we don't really care how much negative health we have, dead is dead
        info['HEALTH'] = max(0.0, info.get('HEALTH', 0.0))
        health = info.get('HEALTH', 0.0) / 30.0
        armor = info.get('ARMOR', 0.0) / 30.0
        kills = info.get('USER2', 0.0) / 10.0  # only works in battle and battle2, this is not really useful

        obs_dict['measurements'].append(selected_weapon)
        obs_dict['measurements'].append(selected_weapon_ammo)
        obs_dict['measurements'].append(health)
        obs_dict['measurements'].append(armor)
        obs_dict['measurements'].append(kills)

        for weapon in range(self.num_weapons):
            obs_dict['measurements'].append(max(0.0, info.get(f'WEAPON{weapon}', 0.0)))
        for weapon in range(self.num_weapons):
            obs_dict['measurements'].append(float(max(0.0, info.get(f'AMMO{weapon}', 0.0))))

        shaping_reward = 0.0

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
                #
                # if abs(reward_delta) > EPS:
                #     log.info('Reward %.3f for %s, delta %.3f (player %r)', reward_delta, var_name, delta, player_id)

                shaping_reward += reward_delta

        # remember new variable values
        for var_name in self.reward_shaping_vars.keys():
            self.prev_vars[var_name] = info.get(var_name, 0.0)

        return obs_dict, shaping_reward

    def reset(self):
        self.prev_vars = {}

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

        # log.info('Original env reward before shaping: %.3f', self._orig_env_reward)
        # player_id = 1
        # if hasattr(self.env.unwrapped, 'player_id'):
        #     player_id = self.env.unwrapped.player_id
        # worker_index = 0
        # if hasattr(self.env.unwrapped, 'worker_index'):
        #     worker_index = self.env.unwrapped.worker_index
        #
        # log.info(
        #     'Total shaping reward is %.3f for %d %d (done %d)',
        #     self._total_shaping_reward, worker_index, player_id, done,
        # )

        return obs_dict, rew, done, info
