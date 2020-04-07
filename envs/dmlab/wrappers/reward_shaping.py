from math import tanh

import gym

from envs.dmlab.dmlab30 import HUMAN_SCORES, RANDOM_SCORES, LEVEL_MAPPING


class DmlabRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_length = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.raw_episode_return += rew
        self.episode_length += info.get('num_frames', 1)

        # optimistic asymmetric clipping from IMPALA paper
        squeezed = tanh(rew / 5.0)
        clipped = 0.3 * squeezed if rew < 0.0 else squeezed
        rew = clipped * 5.0

        if done:
            score = self.raw_episode_return

            level_name = self.unwrapped.level_name
            test_level_name = LEVEL_MAPPING[level_name]

            human = HUMAN_SCORES[test_level_name]
            random = RANDOM_SCORES[test_level_name]
            human_normalized_score = (score - random) / (human - random) * 100
            capped_human_normalized_score = min(100.0, human_normalized_score)

            info['true_reward'] = capped_human_normalized_score
            info['episode_extra_stats'] = dict(
                capped_human_normalized_score=capped_human_normalized_score,
                uncapped_human_normalized_score=human_normalized_score,
                unmodified_score=score,
            )

            level_name_key = f'{self.unwrapped.task_id:02d}_{level_name}'
            info['episode_extra_stats'][f'{level_name_key}_hum_norm_score'] = human_normalized_score
            info['episode_extra_stats'][f'{level_name_key}_len'] = self.episode_length

        return obs, rew, done, info
