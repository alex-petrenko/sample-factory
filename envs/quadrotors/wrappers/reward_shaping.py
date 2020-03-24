import gym


DEFAULT_QUAD_REWARD_SHAPING = dict(
    quad_rewards=dict(
        pos=1.0, pos_linear_weight=1.0, effort=0.05, spin=0.1,
        vel=0.0, crash=1.0, orient=1.0, yaw=0.0,
    ),
)


class QuadsRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, reward_shaping_scheme=None):
        super().__init__(env)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.cumulative_distance_reward = 0.0

        # save a reference to this wrapper in the actual env class, for other wrappers
        self.env.unwrapped._reward_shaping_wrapper = self

    def reset(self):
        obs = self.env.reset()
        self.cumulative_distance_reward = 0.0
        return obs

    def step(self, action):
        # set the (potentially updated) reward shaping scheme
        env_reward_shaping = self.env.unwrapped.rew_coeff
        for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
            env_reward_shaping[key] = weight

        obs, rew, done, info = self.env.step(action)

        rew_dict = info['rewards']
        self.cumulative_distance_reward += rew_dict['rewraw_main']

        if done:
            true_reward = self.cumulative_distance_reward
            info['true_reward'] = true_reward

        return obs, rew, done, info

    def close(self):
        self.env.unwrapped._reward_shaping_wrapper = None
        return self.env.close()
