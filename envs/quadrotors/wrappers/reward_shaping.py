import gym


DEFAULT_QUAD_REWARD_SHAPING = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0,
    ),
)


class QuadsRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, reward_shaping_scheme=None):
        super().__init__(env)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.cumulative_rewards = dict()

        # save a reference to this wrapper in the actual env class, for other wrappers
        self.env.unwrapped._reward_shaping_wrapper = self

    def reset(self):
        obs = self.env.reset()
        self.cumulative_rewards = dict()
        return obs

    def step(self, action):
        # set the (potentially updated) reward shaping scheme
        env_reward_shaping = self.env.unwrapped.rew_coeff
        for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
            env_reward_shaping[key] = weight

        obs, rew, done, info = self.env.step(action)

        rew_dict = info['rewards']

        for key, value in rew_dict.items():
            if key.startswith('rewraw'):
                if key not in self.cumulative_rewards:
                    self.cumulative_rewards[key] = 0
                self.cumulative_rewards[key] += value

        if done:
            true_reward = self.cumulative_rewards['rewraw_main']
            info['true_reward'] = true_reward
            info['episode_extra_stats'] = self.cumulative_rewards

        return obs, rew, done, info

    def close(self):
        self.env.unwrapped._reward_shaping_wrapper = None
        return self.env.close()
