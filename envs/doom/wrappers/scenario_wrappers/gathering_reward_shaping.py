import gym


class DoomGatheringRewardShaping(gym.Wrapper):
    """Reward shaping specific for gathering scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_health = None

    def _reward_shaping(self, info, done):
        if info is None or done:
            return 0.0

        curr_health = info.get('HEALTH', 0.0)
        reward = 0.0

        # min health threshold for which to give reward
        threshold = 10.0

        if self._prev_health is not None:
            delta = curr_health - self._prev_health
            if abs(delta) > threshold:
                reward = 1.0 if delta > 0 else -1

        self._prev_health = curr_health
        return reward

    def reset(self):
        self._prev_health = None
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward += self._reward_shaping(info, done)
        return observation, reward, done, info
