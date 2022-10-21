from typing import Any, Dict, Optional, Tuple, Union

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
        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )


class EnvPoolTo5Tuple(gym.Env):
    r"""A wrapper which can transform an environment from the old API to the new API.
    Old step API refers to step() method returning (observation, reward, done, info), and reset() only retuning the observation.
    New step API refers to step() method returning (observation, reward, terminated, truncated, info) and reset() returning (observation, info).
    (Refer to docs for details on the API change)
    Known limitations:
    - Environments that use `self.np_random` might not work as expected.
    """

    def __init__(self, old_env, render_mode: Optional[str] = None):
        """A wrapper which converts old-style envs to valid modern envs.
        Some information may be lost in the conversion, so we recommend updating your environment.
        Args:
            old_env the env to wrap, implemented with the old API
            render_mode (str): the render mode to use when rendering the environment, passed automatically to env.render
        """
        self.metadata = getattr(old_env, "metadata", {"render_modes": []})
        self.render_mode = render_mode
        self.reward_range = getattr(old_env, "reward_range", None)
        self.spec = getattr(old_env, "spec", None)
        self.env = old_env

        self.observation_space = old_env.observation_space
        self.action_space = old_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets the environment.
        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with
        Returns:
            (observation, info)
        """
        if seed is not None:
            self.seed(seed)
        # Options are ignored

        if self.render_mode == "human":
            self.render()

        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.
        Args:
            action: action to step through the environment with
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        if self.render_mode == "human":
            self.render()

        return convert_to_terminated_truncated_step_api((obs, reward, done, info), is_vector_env=True)

    def render(self) -> Any:
        """Renders the environment.
        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render(mode=self.render_mode)

    def close(self):
        """Closes the environment."""
        self.env.close()

    def __str__(self):
        """Returns the wrapper name and the unwrapped environment string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    def seed(self, _value):
        # The seed is set when the envpool is created, so this does nothing
        ...


def convert_to_terminated_truncated_step_api(
    step_returns: Union[DoneStepType, TerminatedTruncatedStepType], is_vector_env=False
) -> TerminatedTruncatedStepType:
    """Function to transform step returns to new step API irrespective of input API.
    Args:
        step_returns (tuple): Items returned by step(). Can be (obs, rew, done, info) or (obs, rew, terminated, truncated, info)
        is_vector_env (bool): Whether the step_returns are from a vector environment
    """
    if len(step_returns) == 5:
        return step_returns
    else:
        assert len(step_returns) == 4
        observations, rewards, dones, infos = step_returns

        # Cases to handle - info single env /  info vector env (list) / info vector env (dict)
        if is_vector_env is False:
            truncated = infos.pop("TimeLimit.truncated", False)
            return (
                observations,
                rewards,
                dones and not truncated,
                dones and truncated,
                infos,
            )
        elif isinstance(infos, list):
            truncated = np.array([info.pop("TimeLimit.truncated", False) for info in infos])
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        elif isinstance(infos, dict):
            num_envs = len(dones)
            truncated = infos.pop("TimeLimit.truncated", np.zeros(num_envs, dtype=bool))
            return (
                observations,
                rewards,
                np.logical_and(dones, np.logical_not(truncated)),
                np.logical_and(dones, truncated),
                infos,
            )
        else:
            raise TypeError(
                f"Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: {type(infos)}"
            )
