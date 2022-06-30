import numpy as np

import gym

from sample_factory.algo.utils.spaces.discretized import Discretized


class QuadsDiscreteActionsWrapper(gym.Wrapper):
    def __init__(self, env, num_bins):
        super().__init__(env)

        assert isinstance(self.action_space, gym.spaces.Box)
        assert len(self.action_space.shape) == 1

        num_continuous_actions = self.action_space.shape[0]
        spaces = []
        for i in range(num_continuous_actions):
            space = Discretized(num_bins, self.action_space.low[i], self.action_space.high[i])
            spaces.append(space)

        discretized_space = gym.spaces.Tuple(spaces)
        self.action_space = discretized_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        cont_actions = np.ndarray([len(action)], np.float32)
        for i, a in enumerate(action):
            cont_actions[i] = self.action_space.spaces[i].to_continuous(a)

        obs, rew, done, info = self.env.step(cont_actions)

        # log.warning('Min/max state %.2f, %.2f', obs.min(), obs.max())

        return obs, rew, done, info
