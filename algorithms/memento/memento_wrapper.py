import numpy as np

import gym
from gym.spaces import Discrete

from utils.utils import log


class MementoWrapper(gym.core.Wrapper):
    def __init__(self, env, memory_size, memento_increment, memento_decrease):
        super().__init__(env)

        self.mem_max_value = 3.0
        self.memento_increment = memento_increment
        self.memento_decrease = memento_decrease

        # modify the original obs space
        memory_obs_space = gym.spaces.Box(low=-self.mem_max_value, high=self.mem_max_value, shape=[memory_size])

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Dict(spaces=dict(obs=env.observation_space))

        self.observation_space.spaces['memento'] = memory_obs_space

        # modify the original action space
        memory_action_space = Discrete(3 * memory_size)  # three actions for each "memory" cell (noop, inc, dec)

        if isinstance(env.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(list(env.action_space.spaces) + [memory_action_space])
        else:
            self.action_space = gym.spaces.Tuple([env.action_space, memory_action_space])

        self.memory = np.zeros(shape=(memory_size, ), dtype=np.float32)

    def _observation(self, obs):
        if not isinstance(obs, dict):
            obs = dict(obs=obs)

        obs['memento'] = self.memory
        return obs

    def _modify_memory(self, memory_action):
        noop = memory_action % 3 == 0
        if noop:
            return

        memory_cell = memory_action // 3
        increase = memory_action % 3 == 1
        if increase:
            self.memory[memory_cell] += self.memento_increment
        else:
            self.memory[memory_cell] -= self.memento_increment * self.memento_decrease

        self.memory = np.clip(self.memory, -self.mem_max_value, self.mem_max_value)

        # log.info('Memento: %r', self.memory)

    def reset(self):
        self.memory.fill(0.0)
        obs = self._observation(self.env.reset())
        return obs

    def step(self, action):
        memory_action = action[-1]
        self._modify_memory(memory_action)

        action = action[:-1]
        if len(action) == 1:
            action = action[0]

        observation, reward, done, info = self.env.step(action)
        return self._observation(observation), reward, done, info
