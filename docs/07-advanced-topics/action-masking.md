# Action Masking

Action masking is a technique used to restrict the set of actions available to an agent in certain states. This can be particularly useful in environments where some actions are invalid or undesirable in specific situations. See [paper](https://arxiv.org/abs/2006.14171) for more details.

## Implementing Action Masking

To implement action masking in your environment, you need to add an `action_mask` field to the observation dictionary returned by your environment. Here's how to do it:

1. Define the action mask space in your environment's observation space
2. Generate and return the action mask in both `reset()` and `step()` methods

Here's an example of a custom environment implementing action masking:

```python
import gymnasium as gym
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, full_env_name, cfg, render_mode=None):
        ...
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=0, high=1, shape=(3, 3, 2), dtype=np.int8),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
        })
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, **kwargs):
        ...
        # Initial action mask that allows all actions
        action_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        return {"obs": obs, "action_mask": action_mask}, info

    def step(self, action):
        ...
        # Generate new action mask based on the current state
        action_mask = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1])
        return {"obs": obs, "action_mask": action_mask}, reward, terminated, truncated, info
```
