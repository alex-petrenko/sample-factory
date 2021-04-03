import gym
# noinspection PyUnresolvedReferences
import gym_minigrid

from sample_factory.envs.env_wrappers import PixelFormatChwWrapper, RecordingWrapper


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


class MinigridRecordingWrapper(RecordingWrapper):
    def __init__(self, env, record_to):
        super().__init__(env, record_to)

    # noinspection PyMethodOverriding
    def render(self, mode, **kwargs):
        self.env.render()
        frame = self.env.render('rgb_array')
        self._record(frame)
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._recorded_episode_reward += reward
        return observation, reward, done, info


# noinspection PyUnusedLocal
def make_minigrid_env(env_name, cfg=None, **kwargs):
    env = gym.make(env_name)
    env = RenameImageObsWrapper(env)

    if 'record_to' in cfg and cfg.record_to is not None:
        env = MinigridRecordingWrapper(env, cfg.record_to)

    if cfg.pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    return env
