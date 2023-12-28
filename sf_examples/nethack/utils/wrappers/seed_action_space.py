from typing import Any

import gymnasium as gym


class SeedActionSpaceWrapper(gym.Wrapper):
    """
    To have reproducible decorrelate experience we need to seed action space
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.action_space.seed(seed=seed)
        return obs, info
