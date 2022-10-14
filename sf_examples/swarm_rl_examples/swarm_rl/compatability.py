from typing import Any, Dict, Optional, Tuple

import gym
from gym.core import ObsType
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


class QuadEnvCompatibility(gym.Wrapper):
    def __init__(self, env):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            old_env (LegacyEnv): the env to wrap, implemented with the old API
            render_mode (str): the render mode to use when rendering the environment, passed automatically to env.render
        """
        gym.Wrapper.__init__(self, env)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, done, info = self.env.step(action)

        return convert_to_terminated_truncated_step_api((obs, reward, done, info), is_vector_env=True)
