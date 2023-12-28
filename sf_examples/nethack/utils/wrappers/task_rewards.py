import gym

from sf_examples.nethack.utils.task_rewards import (
    EatingScore,
    GoldScore,
    ScoutScore,
    SokobanfillpitScore,
    SokobansolvedlevelsScore,
    StaircasePetScore,
    StaircaseScore,
)


class TaskRewardsInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.tasks = [
            EatingScore(),
            GoldScore(),
            ScoutScore(),
            SokobanfillpitScore(),
            # SokobansolvedlevelsScore(), # TODO: it could have bugs, for now turn off
            StaircasePetScore(),
            StaircaseScore(),
        ]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        for task in self.tasks:
            task.reset_score()

        return obs

    def step(self, action):
        # use tuple and copy to avoid shallow copy (`last_observation` would be the same as `observation`)
        last_observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        obs, reward, done, info = self.env.step(action)
        observation = tuple(a.copy() for a in self.env.unwrapped.last_observation)
        end_status = info["end_status"]

        if done:
            info["episode_extra_stats"] = self.add_more_stats(info)

        # we will accumulate rewards for each step and log them when done signal appears
        for task in self.tasks:
            task.reward(self.env.unwrapped, last_observation, observation, end_status)

        return obs, reward, done, info

    def add_more_stats(self, info):
        extra_stats = info.get("episode_extra_stats", {})
        new_extra_stats = {task.name: task.score for task in self.tasks}
        return {**extra_stats, **new_extra_stats}
