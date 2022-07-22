from typing import Callable

import gym
import numpy as np
import torch
from gym import Wrapper, spaces
from torch import Tensor

from sample_factory.algo.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.algo.utils.tensor_utils import dict_of_lists_cat
from sample_factory.envs.create_env import create_env
from sample_factory.utils.dicts import dict_of_lists_append


class _DictObservationsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.observation_space = gym.spaces.Dict(dict(obs=self.observation_space))


class BatchedDictObservationsWrapper(_DictObservationsWrapper):
    """Guarantees that the environment returns observations as dictionaries of lists (batches)."""

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return dict(obs=obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return dict(obs=obs), rew, done, info


class NonBatchedDictObservationsWrapper(_DictObservationsWrapper):
    """Guarantees that the environment returns observations as lists of dictionaries."""

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, done, info


class TensorWrapper(Wrapper):
    """Ensures that the env returns a dictionary of tensors for observations, and tensors for rewards and dones."""

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents

        self._convert_obs_func = dict()
        self._convert_rew_func = self._convert_dones_func = None

    def _convert(self, obs):
        result = dict()
        for key, value in obs.items():
            result[key] = self._convert_obs_func[key](value)
        return result

    @staticmethod
    def _get_convert_func(x) -> Callable:
        """Depending on type of x, determines the conversion function from x to a tensor."""
        if isinstance(x, torch.Tensor):
            return lambda x_: x_  # do nothing
        elif isinstance(x, np.ndarray):
            return lambda x_: torch.from_numpy(x_)
        elif isinstance(x, (list, tuple)):
            if isinstance(x[0], np.ndarray) or isinstance(x[0], (list, tuple)):
                # creating a tensor from a list of numpy.ndarrays is extremely slow
                # so we first create a numpy array which is then converted to a tensor
                return lambda x_: torch.tensor(np.array(x_))
            elif isinstance(x[0], torch.Tensor):
                return lambda x_: torch.tensor(x_)
            else:
                # just make a tensor and hope for the best
                # leave it like this for now, we can add more cases later if we need to
                return lambda x_: torch.tensor(x_)
        else:
            raise RuntimeError(f"Cannot convert data type {type(x)} to torch.Tensor")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        assert isinstance(obs, dict)

        for key, value in obs.items():
            if key not in self._convert_obs_func:
                self._convert_obs_func[key] = self._get_convert_func(value)

        return self._convert(obs)

    def step(self, action):
        obs, rew, dones, infos = self.env.step(action)
        obs = self._convert(obs)

        if not self._convert_rew_func:
            # the only way to reliably find out the format of data is to actually look what the environment returns
            self._convert_rew_func = self._get_convert_func(rew)
            self._convert_dones_func = self._get_convert_func(dones)

        rew = self._convert_rew_func(rew)
        dones = self._convert_dones_func(dones)
        return obs, rew, dones, infos


class SequentialVectorizeWrapper(Wrapper):
    """Vector interface for multiple environments simulated sequentially on one worker."""

    def __init__(self, envs):
        super().__init__(envs[0])
        self.single_env_agents = envs[0].num_agents
        assert all(
            e.num_agents == self.single_env_agents for e in envs
        ), f"Expect all envs to have the same number of agents {self.single_env_agents}"

        self.envs = envs
        self.num_agents = self.single_env_agents * len(envs)

        self.obs = None
        self.rew = None
        self.dones = None
        self.infos = None

    def reset(self, **kwargs):
        self.obs = dict()
        for e in self.envs:
            dict_of_lists_append(self.obs, e.reset(**kwargs))

        dict_of_lists_cat(self.obs)
        return self.obs

    def step(self, actions: Tensor):
        infos = []
        ofs = 0
        next_ofs = self.single_env_agents
        for i, e in enumerate(self.envs):
            idx = slice(ofs, next_ofs)
            env_actions = actions[idx]
            obs, rew, dones, info = e.step(env_actions)

            # TODO: test if this works for multi-agent envs
            for key, x in obs.items():
                self.obs[key][idx] = x

            if self.rew is None:
                self.rew = rew.repeat(len(self.envs))
                self.dones = dones.repeat(len(self.envs))

            self.rew[idx] = rew
            self.dones[idx] = dones

            infos.extend(info)

            ofs += self.single_env_agents
            next_ofs += self.single_env_agents

        return self.obs, self.rew, self.dones, infos

    def close(self):
        for e in self.envs:
            e.close()


def _make_env_func(cfg, env_config) -> gym.Env:
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not is_multiagent_env(env):
        env = MultiAgentWrapper(env)
    return env


def make_env_func_batched(cfg, env_config) -> gym.Env:
    """
    This should yield an environment that always returns a dict of PyTorch tensors (CPU- or GPU-side) or
    a dict of numpy arrays or a dict of lists (depending on what the environment returns in the first place).
    """
    env = _make_env_func(cfg, env_config)
    if not isinstance(env.observation_space, spaces.Dict):
        env = BatchedDictObservationsWrapper(env)

    # At this point we can be sure that our environment outputs a dictionary of lists (or numpy arrays or tensors)
    # containing obs, rewards, etc. for each agent in the environment. If it wasn't true to begin with, we guaranteed
    # that by adding wrappers above.
    # Now we just want the environment to return a tensor dict for observations and tensors for rewards and dones.
    # We leave infos intact for now, because format of infos isn't really specified and can be inconsistent between
    # timesteps.
    env = TensorWrapper(env)
    return env


def make_env_func_non_batched(cfg, env_config) -> gym.Env:
    """
    This should yield an environment that always returns a list of everything (list of dict observations, rewards,
    dones, etc.)
    This is for the non-batched sampler which processes each agent's data independently without any vectorization
    attempts (and therefore enables more sophisticated configurations where agents in the same env can be controlled
    by different policies).
    """
    env = _make_env_func(cfg, env_config)
    if not isinstance(env.observation_space, spaces.Dict):
        env = NonBatchedDictObservationsWrapper(env)
    return env
