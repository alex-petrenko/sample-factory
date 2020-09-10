import gym
import numpy as np


DEFAULT_QUAD_REWARD_SHAPING = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0, quadcol_bin=0., quadsettle=0.,
    ),
)


class QuadsRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, reward_shaping_scheme=None):
        super().__init__(env)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.cumulative_rewards = None
        self.episode_actions = None

        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else 1

        # save a reference to this wrapper in the actual env class, for other wrappers
        self.env.unwrapped._reward_shaping_wrapper = self

    def reset(self):
        obs = self.env.reset()
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]
        self.episode_actions = []
        return obs

    def step(self, action):
        self.episode_actions.append(action)

        # set the (potentially updated) reward shaping scheme
        env_reward_shaping = self.env.unwrapped.rew_coeff
        for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
            env_reward_shaping[key] = weight

        obs, rewards, dones, infos = self.env.step(action)
        if self.num_agents == 1:
            infos_multi, dones_multi = [infos], [dones]
        else:
            infos_multi, dones_multi = infos, dones

        for i, info in enumerate(infos_multi):
            rew_dict = info['rewards']

            for key, value in rew_dict.items():
                if key.startswith('rewraw'):
                    if key not in self.cumulative_rewards[i]:
                        self.cumulative_rewards[i][key] = 0
                    self.cumulative_rewards[i][key] += value

            if dones_multi[i]:
                true_reward = self.cumulative_rewards[i]['rewraw_main']
                info['true_reward'] = true_reward
                info['episode_extra_stats'] = self.cumulative_rewards[i]

                episode_actions = np.array(self.episode_actions)
                episode_actions = episode_actions.transpose()
                for action_idx in range(episode_actions.shape[0]):
                    mean_action = np.mean(episode_actions[action_idx])
                    std_action = np.std(episode_actions[action_idx])
                    info['episode_extra_stats'][f'z_action{action_idx}_mean'] = mean_action
                    info['episode_extra_stats'][f'z_action{action_idx}_std'] = std_action

                self.cumulative_rewards[i] = dict()

        if any(dones_multi):
            self.episode_actions = []

        return obs, rewards, dones, infos

    def close(self):
        self.env.unwrapped._reward_shaping_wrapper = None
        return self.env.close()
