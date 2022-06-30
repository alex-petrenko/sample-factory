import copy

import gym
import numpy as np

from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface


DEFAULT_QUAD_REWARD_SHAPING_SINGLE = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0,
    ),
)

DEFAULT_QUAD_REWARD_SHAPING = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING_SINGLE)
DEFAULT_QUAD_REWARD_SHAPING['quad_rewards'].update(dict(
    quadcol_bin=0.0, quadcol_bin_obst=0.0, quadsettle=0.0,
))


class QuadsRewardShapingWrapper(gym.Wrapper, RewardShapingInterface, TrainingInfoInterface):
    def __init__(self, env, reward_shaping_scheme=None, annealing=None):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)
        TrainingInfoInterface.__init__(self)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.cumulative_rewards = None
        self.episode_actions = None

        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else 1

        self.reward_shaping_updated = True

        self.annealing = annealing

    def get_default_reward_shaping(self):
        return self.reward_shaping_scheme

    def get_current_reward_shaping(self, agent_idx: int):
        return self.reward_shaping_scheme

    def set_reward_shaping(self, reward_shaping, unused_agent_idx):
        self.reward_shaping_scheme = reward_shaping
        self.reward_shaping_updated = True

    def reset(self):
        obs = self.env.reset()
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]
        self.episode_actions = []
        return obs

    def step(self, action):
        self.episode_actions.append(action)

        if self.reward_shaping_updated:
            # set the updated reward shaping scheme
            env_reward_shaping = self.env.unwrapped.rew_coeff
            for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
                env_reward_shaping[key] = weight

            self.reward_shaping_updated = False

        obs, rewards, dones, infos = self.env.step(action)
        if self.env.is_multiagent:
            infos_multi, dones_multi = infos, dones
        else:
            infos_multi, dones_multi = [infos], [dones]

        for i, info in enumerate(infos_multi):
            rew_dict = info['rewards']

            for key, value in rew_dict.items():
                if key.startswith('rew'):
                    if key not in self.cumulative_rewards[i]:
                        self.cumulative_rewards[i][key] = 0
                    self.cumulative_rewards[i][key] += value

            if dones_multi[i]:
                true_reward = self.cumulative_rewards[i]['rewraw_main']
                true_reward_consider_collisions = True
                if true_reward_consider_collisions:
                    # we ideally want zero collisions, so collisions between quads are given very high weight
                    true_reward += 1000 * self.cumulative_rewards[i].get('rewraw_quadcol', 0)

                info['true_reward'] = true_reward
                if 'episode_extra_stats' not in info:
                    info['episode_extra_stats'] = dict()
                extra_stats = info['episode_extra_stats']
                extra_stats.update(self.cumulative_rewards[i])

                approx_total_training_steps = self.training_info.get('approx_total_training_steps', 0)
                extra_stats['z_approx_total_training_steps'] = approx_total_training_steps

                if hasattr(self.env.unwrapped, 'scenario') and self.env.unwrapped.scenario:
                    scenario_name = self.env.unwrapped.scenario.name()
                    for rew_key in ['rew_pos', 'rewraw_pos', 'rew_crash', 'rewraw_crash']:
                        extra_stats[f'{rew_key}_{scenario_name}'] = self.cumulative_rewards[i][rew_key]

                episode_actions = np.array(self.episode_actions)
                episode_actions = episode_actions.transpose()
                for action_idx in range(episode_actions.shape[0]):
                    mean_action = np.mean(episode_actions[action_idx])
                    std_action = np.std(episode_actions[action_idx])
                    extra_stats[f'z_action{action_idx}_mean'] = mean_action
                    extra_stats[f'z_action{action_idx}_std'] = std_action

                self.cumulative_rewards[i] = dict()

                if self.annealing:
                    env_reward_shaping = self.env.unwrapped.rew_coeff
                    # annealing from 0.0 to final value
                    for anneal_schedule in self.annealing:
                        coeff_name = anneal_schedule.coeff_name
                        final_value = anneal_schedule.final_value
                        anneal_steps = anneal_schedule.anneal_env_steps
                        env_reward_shaping[coeff_name] = min(final_value * approx_total_training_steps / anneal_steps, final_value)
                        extra_stats[f'z_anneal_{coeff_name}'] = env_reward_shaping[coeff_name]

        if any(dones_multi):
            self.episode_actions = []

        return obs, rewards, dones, infos
