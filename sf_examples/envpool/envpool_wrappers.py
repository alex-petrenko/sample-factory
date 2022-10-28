from typing import Tuple, Union

import gym
import numpy as np
from gym.core import ObsType

DoneStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]

TerminatedTruncatedStepType = Tuple[
    Union[ObsType, np.ndarray],
    Union[float, np.ndarray],
    Union[bool, np.ndarray],
    Union[bool, np.ndarray],
    Union[dict, list],
]


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class EnvPoolResetFixWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        needs_reset = np.nonzero(terminated | truncated)[0]
        obs[needs_reset], _ = self.env.reset(needs_reset)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # envpool does not support the seed in reset, even with the updated API
        return super().reset(**kwargs)


class BatchedRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, num_envs, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", num_envs)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, infos = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - terminated
        self.episode_lengths *= 1 - terminated
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, terminated, truncated, infos
