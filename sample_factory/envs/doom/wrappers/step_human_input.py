import numpy as np

import gym


# noinspection PyProtectedMember
class StepHumanInput(gym.Wrapper):
    """Doom wrapper that allows to play with human input."""

    def __init__(self, env):
        super(StepHumanInput, self).__init__(env)

    def reset(self):
        self.unwrapped.mode = 'human'
        self.unwrapped._ensure_initialized()
        return self.env.reset()

    def step(self, action):
        del action  # we actually ignore action and take input from keyboard

        self.unwrapped.mode = 'human'
        self.unwrapped._ensure_initialized()

        doom_env = self.unwrapped
        doom_env.game.advance_action()

        state = doom_env.game.get_state()
        done = doom_env.game.is_episode_finished()
        reward = doom_env.game.get_last_reward()

        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {'dummy': 0}

        return observation, reward, done, info
