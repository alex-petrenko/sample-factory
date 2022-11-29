from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch
from gym import Wrapper, spaces
from gym.core import ActType, ObsType
from torch import Tensor

from sample_factory.algo.utils.tensor_utils import dict_of_lists_cat
from sample_factory.envs.create_env import create_env
from sample_factory.envs.env_utils import (
    RewardShapingInterface,
    TrainingInfoInterface,
    find_training_info_interface,
    find_wrapper_interface,
)
from sample_factory.utils.dicts import dict_of_lists_append, list_of_dicts_to_dict_of_lists
from sample_factory.utils.typing import Config

Actions = Any
ListActions = Sequence[Actions]
TensorActions = Tensor

SeqBools = Sequence[bool]

DictObservations = Dict[str, Any]
DictOfListsObservations = Dict[str, Sequence[Any]]
DictOfTensorObservations = Dict[str, Tensor]
ListObservations = Sequence[Any]
ListOfDictObservations = Sequence[DictObservations]


def get_multiagent_info(env: Any) -> Tuple[bool, int]:
    num_agents = env.num_agents if hasattr(env, "num_agents") else 1
    is_multiagent = env.is_multiagent if hasattr(env, "is_multiagent") else num_agents > 1
    assert is_multiagent or num_agents == 1, f"Invalid configuration: {is_multiagent=} and {num_agents=}"
    return is_multiagent, num_agents


def is_multiagent_env(env: Any) -> bool:
    is_multiagent, num_agents = get_multiagent_info(env)
    return is_multiagent


class _DictObservationsWrapper(Wrapper[ObsType, ActType]):
    def __init__(self, env):
        super().__init__(env)
        is_multiagent, num_agents = get_multiagent_info(env)
        self.is_multiagent: bool = is_multiagent
        self.num_agents: int = num_agents
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(dict(obs=self.observation_space))


class BatchedDictObservationsWrapper(_DictObservationsWrapper[DictObservations, Actions]):
    """Guarantees that the environment returns observations as dictionaries of lists (batches)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return dict(obs=obs), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return dict(obs=obs), rew, terminated, truncated, info


class BatchedMultiAgentWrapper(Wrapper[DictOfListsObservations, Actions]):
    """Assumes wrapped environment has dictionary obs space."""

    def __init__(self, env):
        super().__init__(env)
        self.num_agents: int = 1
        self.is_multiagent: bool = True
        assert isinstance(env.observation_space, spaces.Dict), "Wrapped environment must have dictionary obs space."

        self.obs_dict = {}

    def _obs(self, obs: Dict) -> DictOfListsObservations:
        for key, value in obs.items():
            self.obs_dict[key] = [value]
        return self.obs_dict

    def reset(self, **kwargs) -> Tuple[DictOfListsObservations, Sequence[Dict]]:
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), [info]

    def step(self, action) -> Tuple[DictOfListsObservations, Sequence, SeqBools, SeqBools, Sequence[Dict]]:
        action = action[0]
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated | truncated:  # auto-resetting
            obs, info["reset_info"] = self.env.reset()
        return self._obs(obs), [rew], [terminated], [truncated], [info]


class NonBatchedMultiAgentWrapper(Wrapper[ListObservations, ListActions]):
    """
    This wrapper allows us to treat a single-agent environment as multi-agent with 1 agent.
    That is, the data (obs, rewards, etc.) is converted into lists of length 1
    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents: int = 1
        self.is_multiagent: bool = True

    def reset(self, **kwargs) -> ListObservations:
        obs, info = self.env.reset(**kwargs)
        return [obs], [info]

    def step(self, action: ListActions) -> Tuple[ListObservations, Sequence, SeqBools, SeqBools, Sequence[Dict]]:
        action = action[0]
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:  # auto-resetting
            obs, info["reset_info"] = self.env.reset()
        return [obs], [rew], [terminated], [truncated], [info]


class NonBatchedDictObservationsWrapper(_DictObservationsWrapper[ListOfDictObservations, ListActions]):
    """Guarantees that the environment returns observations as lists of dictionaries."""

    def reset(self, **kwargs) -> ListOfDictObservations:
        obs, info = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs], info

    def step(self, action: ListActions) -> Tuple[ListOfDictObservations, Any, Any, Any, Any]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, terminated, truncated, info


class BatchedListToDictWrapper(Wrapper[DictOfListsObservations, Actions]):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, list):
            return list_of_dicts_to_dict_of_lists(obs), info
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, list):
            return list_of_dicts_to_dict_of_lists(obs), rew, terminated, truncated, info
        return obs, rew, terminated, truncated, info


class BatchedVecEnv(Wrapper[DictOfTensorObservations, TensorActions]):
    """Ensures that the env returns a dictionary of tensors for observations, and tensors for rewards and dones."""

    ConvertFunc = Callable[[Any], Tensor]

    def __init__(self, env):
        if not isinstance(env.observation_space, spaces.Dict):
            env = BatchedDictObservationsWrapper(env)
        if not is_multiagent_env(env):
            env = BatchedMultiAgentWrapper(env)
        else:
            env = BatchedListToDictWrapper(env)

        is_multiagent, num_agents = get_multiagent_info(env)
        self.is_multiagent: bool = is_multiagent
        self.num_agents: int = num_agents

        self._convert_obs_func: Dict[str, BatchedVecEnv.ConvertFunc] = dict()
        self._convert_rew_func = self._convert_terminated_func = self._convert_truncated_func = None

        self._seed: Optional[int] = None
        self._seeded: bool = False

        super().__init__(env)

    def _convert(self, obs: Dict[str, Any]) -> DictOfTensorObservations:
        result = dict()
        for key, value in obs.items():
            result[key] = self._convert_obs_func[key](value)
        return result

    @staticmethod
    def _get_convert_func(x: Union[Tensor, np.ndarray, List, Tuple]) -> ConvertFunc:
        """Depending on type of x, determines the conversion function from x to a tensor."""
        if isinstance(x, Tensor):
            return lambda x_: x_  # do nothing
        elif isinstance(x, np.ndarray):
            return lambda x_: torch.from_numpy(x_)
        elif isinstance(x, (list, tuple)):
            if isinstance(x[0], np.ndarray) or isinstance(x[0], (list, tuple)):
                # creating a tensor from a list of numpy.ndarrays is extremely slow
                # so we first create a numpy array which is then converted to a tensor
                return lambda x_: torch.tensor(np.array(x_))
            elif isinstance(x[0], Tensor):
                return lambda x_: torch.tensor(x_)
            else:
                # just make a tensor and hope for the best
                # leave it like this for now, we can add more cases later if we need to
                return lambda x_: torch.tensor(x_)
        else:
            raise RuntimeError(f"Cannot convert data type {type(x)} to torch.Tensor")

    def seed(self, seed: Optional[int] = None):
        """
        Since Gym 0.26 seeding is done in reset().
        Sample Factory uses its own wrappers around gym.Env so we just keep this function and forward the seed to
        the first reset() if needed.
        """
        self._seed = seed

    def reset(self, **kwargs) -> Tuple[DictOfTensorObservations, Dict]:
        if not self._seeded and self._seed is not None:
            kwargs["seed"] = self._seed
            self._seeded = True

        obs, info = self.env.reset(**kwargs)
        assert isinstance(obs, dict)

        for key, value in obs.items():
            if key not in self._convert_obs_func:
                self._convert_obs_func[key] = self._get_convert_func(value)

        return self._convert(obs), info

    def step(self, action: TensorActions) -> Tuple[DictOfTensorObservations, Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        obs = self._convert(obs)

        if not self._convert_rew_func:
            # the only way to reliably find out the format of data is to actually look what the environment returns
            # noinspection PyTypeChecker
            self._convert_rew_func = self._get_convert_func(rew)
            # noinspection PyTypeChecker
            self._convert_terminated_func = self._get_convert_func(terminated)
            # noinspection PyTypeChecker
            self._convert_truncated_func = self._get_convert_func(truncated)

        rew = self._convert_rew_func(rew)
        terminated = self._convert_terminated_func(terminated)
        truncated = self._convert_truncated_func(truncated)
        return obs, rew, terminated, truncated, infos


class SequentialVectorizeWrapper(Wrapper, TrainingInfoInterface, RewardShapingInterface):
    """Vector interface for multiple environments simulated sequentially on one worker."""

    def __init__(self, envs: Sequence):
        Wrapper.__init__(self, envs[0])
        TrainingInfoInterface.__init__(self)
        self.single_env_agents = envs[0].num_agents
        assert all(
            e.num_agents == self.single_env_agents for e in envs
        ), f"Expect all envs to have the same number of agents {self.single_env_agents}"

        self.envs = envs
        self.num_agents = self.single_env_agents * len(envs)

        self.obs = self.rew = self.terminated = self.truncated = self.infos = None

        self.training_info_interfaces: Optional[List[TrainingInfoInterface]] = []
        self.reward_shaping_interfaces: Optional[List[RewardShapingInterface]] = []
        for env in envs:
            env_train_info = find_training_info_interface(env)
            if env_train_info is None:
                self.training_info_interfaces = None
                break
            else:
                self.training_info_interfaces.append(env_train_info)

            env_rew_shaping = find_wrapper_interface(env, RewardShapingInterface)
            if env_rew_shaping is None:
                self.reward_shaping_interfaces = None
                break
            else:
                self.reward_shaping_interfaces.append(env_rew_shaping)

    def reset(self, **kwargs) -> Tuple[Dict, List[Dict]]:
        infos = []
        self.obs = dict()
        for e in self.envs:
            obs, info = e.reset(**kwargs)
            dict_of_lists_append(self.obs, obs)
            infos.extend(info)

        dict_of_lists_cat(self.obs)
        return self.obs, infos

    def step(self, actions: Tensor):
        infos = []
        ofs = 0
        next_ofs = self.single_env_agents
        for i, e in enumerate(self.envs):
            idx = slice(ofs, next_ofs)
            env_actions = actions[idx]
            obs, rew, terminated, truncated, info = e.step(env_actions)

            # TODO: test if this works for multi-agent envs
            for key, x in obs.items():
                self.obs[key][idx] = x

            if self.rew is None:
                self.rew = rew.repeat(len(self.envs))
                self.terminated = terminated.repeat(len(self.envs))
                self.truncated = truncated.repeat(len(self.envs))

            self.rew[idx] = rew
            self.terminated[idx] = terminated
            self.truncated[idx] = truncated

            infos.extend(info)

            ofs += self.single_env_agents
            next_ofs += self.single_env_agents

        return self.obs, self.rew, self.terminated, self.truncated, infos

    def set_training_info(self, training_info: Dict) -> None:
        for env_train_info in self.training_info_interfaces:
            env_train_info.set_training_info(training_info)

    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        if self.reward_shaping_interfaces is not None:
            return self.reward_shaping_interfaces[0].get_default_reward_shaping()
        else:
            return None

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_indices: int | slice) -> None:
        assert isinstance(agent_indices, slice)
        for agent_idx in range(agent_indices.start, agent_indices.stop):
            env_idx = agent_idx // self.single_env_agents
            env_agent_idx = agent_idx % self.single_env_agents
            self.reward_shaping_interfaces[env_idx].set_reward_shaping(reward_shaping, env_agent_idx)

    def close(self):
        for e in self.envs:
            e.close()


def make_env_func_batched(cfg, env_config, render_mode: Optional[str] = None) -> BatchedVecEnv:
    """
    This should yield an environment that always returns a dict of PyTorch tensors (CPU- or GPU-side) or
    a dict of numpy arrays or a dict of lists (depending on what the environment returns in the first place).
    """
    env = create_env(cfg.env, cfg=cfg, env_config=env_config, render_mode=render_mode)

    # At this point we can be sure that our environment outputs a dictionary of lists (or numpy arrays or tensors)
    # containing obs, rewards, etc. for each agent in the environment.
    # Now we just want the environment to return a tensor dict for observations and tensors for rewards and dones.
    # We leave infos intact for now, because format of infos isn't really specified and can be inconsistent between
    # timesteps.
    env = BatchedVecEnv(env)
    return env


class NonBatchedVecEnv(Wrapper[ListObservations, ListActions]):
    """
    reset() returns a list of dict observations.
    step(action) returns a list of dict observations, list of rewards, list of dones, list of infos.
    """

    def __init__(self, env):
        if not is_multiagent_env(env):
            env = NonBatchedMultiAgentWrapper(env)
        if not isinstance(env.observation_space, spaces.Dict):
            env = NonBatchedDictObservationsWrapper(env)

        is_multiagent, num_agents = get_multiagent_info(env)
        self.is_multiagent: bool = is_multiagent
        self.num_agents: int = num_agents
        super().__init__(env)


def make_env_func_non_batched(cfg: Config, env_config, render_mode: Optional[str] = None) -> NonBatchedVecEnv:
    """
    This should yield an environment that always returns a list of {observations, rewards,
    dones, etc.}
    This is for the non-batched sampler which processes each agent's data independently without any vectorization
    (and therefore enables more sophisticated configurations where agents in the same env can be controlled
    by different policies and so on).
    """
    env = create_env(cfg.env, cfg=cfg, env_config=env_config, render_mode=render_mode)
    env = NonBatchedVecEnv(env)
    return env
