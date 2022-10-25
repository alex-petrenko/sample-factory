import gym


class DoomGatheringRewardShaping(gym.Wrapper):
    """Reward shaping specific for gathering scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_health = None
        self.orig_env_reward = 0.0

    def _reward_shaping(self, info, done):
        if info is None or done:
            return 0.0

        curr_health = info.get("HEALTH", 0.0)
        reward = 0.0

        if self._prev_health is not None:
            delta = curr_health - self._prev_health
            if delta > 0.0:
                reward = 1.0

        self._prev_health = curr_health
        return reward

    def reset(self, **kwargs):
        self._prev_health = None
        self.orig_env_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.orig_env_reward += reward

        done = terminated | truncated
        reward += self._reward_shaping(info, done)

        if done:
            true_objective = self.orig_env_reward
            info["true_objective"] = true_objective

        return observation, reward, terminated, truncated, info
