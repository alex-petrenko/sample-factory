import gym
import numpy as np


# noinspection PyProtectedMember
class StepHumanInput(gym.Wrapper):
    """Doom wrapper that allows to play with human input."""

    def __init__(self, env):
        super(StepHumanInput, self).__init__(env)

    def reset(self, **kwargs):
        self.unwrapped.mode = "human"
        self.unwrapped._ensure_initialized()
        return self.env.reset(**kwargs)

    def step(self, action):
        del action  # we actually ignore action and take input from keyboard

        self.unwrapped.mode = "human"
        self.unwrapped._ensure_initialized()

        doom_env = self.unwrapped
        doom_env.game.advance_action()

        state = doom_env.game.get_state()
        reward = doom_env.game.get_last_reward()
        terminated = doom_env.game.is_episode_finished()
        truncated = False

        if not terminated:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.uint8(np.zeros(self.observation_space.shape))

        info = {"dummy": 0}
        return observation, reward, terminated, truncated, info
