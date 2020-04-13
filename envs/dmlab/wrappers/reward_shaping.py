from math import tanh

import gym

RAW_SCORE_SUMMARY_KEY_SUFFIX = 'dmlab_raw_score'


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

            info['episode_extra_stats'] = dict()
            level_name = self.unwrapped.level_name

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.unwrapped.task_id:02d}_{level_name}'
            info['episode_extra_stats'][f'{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}'] = score
            info['episode_extra_stats'][f'{level_name_key}_len'] = self.episode_length

        return obs, rew, done, info
