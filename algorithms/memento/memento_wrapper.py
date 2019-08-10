from collections import deque

import gym
import numpy as np
from gym.spaces import Discrete

from utils.utils import AttrDict


def get_memento_args(params):
    memento_args = AttrDict(dict(
        memento_size=params.memento_size,
        memento_increment=params.memento_increment,
        memento_history=params.memento_history,
    ))
    return memento_args


class MementoWrapper(gym.core.Wrapper):
    def __init__(self, env, memento_args):
        super().__init__(env)

        self.mem_max_value = 3.0
        self.memory_size = memento_args.memento_size
        self.increment = memento_args.memento_increment
        self.history = memento_args.memento_history

        # modify the original obs space
        memory_obs_size = self.memory_size * self.history + self.memory_size * self.history
        memory_obs_space = gym.spaces.Box(low=-self.mem_max_value, high=self.mem_max_value, shape=[memory_obs_size])

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Dict(spaces=dict(obs=env.observation_space))

        self.observation_space.spaces['memento'] = memory_obs_space

        # three actions for each "memory" cell (noop, inc, dec)
        memory_action_spaces = [Discrete(3) for _ in range(self.memory_size)]

        # modify the original action space
        if isinstance(env.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(list(env.action_space.spaces) + memory_action_spaces)
        else:
            self.action_space = gym.spaces.Tuple([env.action_space] + memory_action_spaces)

        self.memory = self.past_memory = self.past_actions = None
        self._reset_memory()

    def _reset_memory(self):
        self.memory = np.zeros(shape=(self.memory_size,), dtype=np.float32)
        self.past_memory = deque([np.zeros(shape=(self.memory_size,))] * self.history, maxlen=self.history)
        self.past_actions = deque([np.zeros(shape=(self.memory_size,))] * self.history, maxlen=self.history)

    def _observation(self, obs):
        if not isinstance(obs, dict):
            obs = dict(obs=obs)

        obs['memento'] = np.array(list(self.past_memory) + list(self.past_actions)).flatten()
        return obs

    def _modify_memory(self, memory_actions):
        memory_actions = np.asarray(memory_actions) - 1
        for memory_cell, memory_action in enumerate(memory_actions):
            # actions:
            # 0 = noop
            # +1 = increase cell value
            # -1 = decrease cell value

            self.memory[memory_cell] += self.increment * memory_action

        self.memory = np.clip(self.memory, -self.mem_max_value, self.mem_max_value)
        self.past_memory.append(self.memory.copy())
        self.past_actions.append(memory_actions)

        # log.info('Memento: %r', self.memory)

    def reset(self):
        self._reset_memory()
        obs = self._observation(self.env.reset())
        return obs

    def step(self, action):
        memory_action = action[-self.memory_size:]
        self._modify_memory(memory_action)

        action = action[:-self.memory_size]
        if len(action) == 1:
            action = action[0]

        observation, reward, done, info = self.env.step(action)
        return self._observation(observation), reward, done, info
