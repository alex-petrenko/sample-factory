from math import tanh

import gym

from algorithms.utils.algo_utils import EPS
from envs.dmlab.dmlab30 import HUMAN_SCORES, RANDOM_SCORES


class DmlabRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_episode_return = 0

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.raw_episode_return += rew

        # optimistic asymmetric clipping from IMPALA paper
        squeezed = tanh(rew / 5.0)
        clipped = 0.3 * squeezed if rew < 0.0 else squeezed
        rew = clipped * 5.0

        if done:
            score = self.raw_episode_return

            level_name = self.unwrapped.level_name
            human = HUMAN_SCORES[level_name]
            random = RANDOM_SCORES[level_name]
            human_normalized_score = (score - random) / (human - random + EPS) * 100
            capped_human_normalized_score = max(100.0, human_normalized_score)

            info['true_reward'] = capped_human_normalized_score
            info['episode_extra_stats'] = dict(
                capped_human_normalized_score=capped_human_normalized_score,
                uncapped_human_normalized_score=human_normalized_score,
                unmodified_score=score,
            )

        return obs, rew, done, info
