import gym

# noinspection PyUnresolvedReferences
import gym_minigrid

from envs.env_wrappers import PixelFormatChwWrapper


class RenameImageObsWrapper(gym.ObservationWrapper):
    """We call the main observation just 'obs' in all algorithms."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space
        self.observation_space.spaces['obs'] = self.observation_space.spaces['image']
        self.observation_space.spaces.pop('image')

    def observation(self, observation):
        observation['obs'] = observation['image']
        observation.pop('image')
        return observation


# noinspection PyUnusedLocal
def make_minigrid_env(env_name, pixel_format='HWC', **kwargs):
    env = gym.make(env_name)
    env = RenameImageObsWrapper(env)

    if pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    return env
