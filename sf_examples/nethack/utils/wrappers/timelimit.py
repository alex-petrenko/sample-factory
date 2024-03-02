import gym
from nle.env.base import NLE


class NLETimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info["TimeLimit.truncated"] = True if info["end_status"] == NLE.StepStatus.ABORTED else False
        return obs, reward, done, info
