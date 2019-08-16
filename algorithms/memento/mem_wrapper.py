import numpy as np

import gym
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from utils.utils import log


class MemActionSpace(Discrete):
    def __init__(self, n, prior_probs):
        super().__init__(n)

        self.prior_probs = prior_probs
        assert n == len(self.prior_probs)


# noinspection PyAbstractClass
class MemCategorical(Categorical):
    def __init__(self, logits, prior_probs):
        super().__init__(logits=logits)
        self.prior_probs = torch.tensor(prior_probs, device=logits.device)

    def entropy(self):
        """Actually return KL-divergence between this distribution and the prior."""
        probs = self.probs
        prob_ratios = probs / self.prior_probs
        kl = probs * torch.log(prob_ratios)
        kl = kl.sum(dim=-1)

        # negate, because increasing entropy in this case is the same as decreasing KL-divergence
        return -kl


class MemWrapper(gym.core.Wrapper):
    def __init__(self, env, mem_size, mem_feature):
        super().__init__(env)

        self.mem_size = mem_size
        self.mem_feature = mem_feature

        memory_obs_size = self.mem_size * self.mem_feature
        memory_obs_space = gym.spaces.Box(low=-1e3, high=1e3, shape=[memory_obs_size])

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Dict(spaces=dict(obs=env.observation_space))

        self.observation_space.spaces['mem'] = memory_obs_space

        # two actions for each "memory" cell (noop, rewrite)
        memory_action_spaces = []
        priors = [0.5, 0.1, 0.05, 0.01]
        for i in range(self.mem_size):
            write_probability = priors[i] if i < len(priors) else 0.1
            memory_action_space = MemActionSpace(2, prior_probs=[1.0 - write_probability, write_probability])
            memory_action_spaces.append(memory_action_space)

        # modify the original action space
        if isinstance(env.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(list(env.action_space.spaces) + memory_action_spaces)
        else:
            self.action_space = gym.spaces.Tuple([env.action_space] + memory_action_spaces)

        self.memory = None
        self.memory_empty = True
        self._reset_memory()

    def _reset_memory(self):
        self.memory = np.zeros(shape=(self.mem_size, self.mem_feature), dtype=np.float32)
        self.memory_empty = True

    def _parse_actions(self, actions):
        num_memory_actions = self.mem_size + self.mem_feature
        env_actions = actions[:-num_memory_actions]
        memory_actions_and_state = actions[-num_memory_actions:]

        memory_actions = memory_actions_and_state[:self.mem_size]
        new_state = memory_actions_and_state[self.mem_size:]

        assert len(new_state) == self.mem_feature

        return env_actions, memory_actions, new_state

    def _modify_memory(self, memory_actions, agent_state):
        for cell_idx in range(self.mem_size):
            if memory_actions[cell_idx] != 0 or self.memory_empty:
                self.memory[cell_idx] = agent_state

        self.memory_empty = False

    def _observation(self, obs):
        if not isinstance(obs, dict):
            obs = dict(obs=obs)
        obs['mem'] = np.array(self.memory).flatten()
        return obs

    def reset(self):
        self._reset_memory()
        obs = self._observation(self.env.reset())
        return obs

    def step(self, actions):
        env_actions, memory_actions, agent_state = self._parse_actions(actions)
        # log.info('Memory actions: %r', list(memory_actions))
        self._modify_memory(memory_actions, agent_state)

        if len(env_actions) == 1:
            env_actions = env_actions[0]

        observation, reward, done, info = self.env.step(env_actions)
        return self._observation(observation), reward, done, info
