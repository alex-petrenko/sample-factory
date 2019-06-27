import gym

resolutions = ['160x120', '200x125', '200x150', '256x144', '256x160', '256x192', '320x180', '320x200',
               '320x240', '320x256', '400x225', '400x250', '400x300', '512x288', '512x320', '512x384',
               '640x360', '640x400', '640x480', '800x450', '800x500', '800x600', '1024x576', '1024x640',
               '1024x768', '1280x720', '1280x800', '1280x960', '1280x1024', '1400x787', '1400x875',
               '1400x1050', '1600x900', '1600x1000', '1600x1200', '1920x1080']


class SetResolutionWrapper(gym.Wrapper):
    """Doom wrapper to change screen resolution."""

    def __init__(self, env, target_resolution):
        super(SetResolutionWrapper, self).__init__(env)
        if target_resolution not in resolutions:
            raise gym.error.Error(
                'Error - The specified resolution "{}" is not supported by Vizdoom.'.format(target_resolution),
            )

        orig_obs_space = self.observation_space

        parts = target_resolution.lower().split('x')
        width = int(parts[0])
        height = int(parts[1])
        screen_res = __import__('vizdoom')
        screen_res = getattr(screen_res, 'ScreenResolution')
        screen_res = getattr(screen_res, 'RES_{}X{}'.format(width, height))

        self.unwrapped.screen_w = width
        self.unwrapped.screen_h = height
        self.unwrapped.screen_resolution = screen_res
        self.unwrapped.calc_observation_space()

        if isinstance(orig_obs_space, gym.spaces.Dict):
            new_obs_space = {}
            for key, value in orig_obs_space.spaces.items():
                new_obs_space[key] = self.unwrapped.observation_space
            new_obs_space = gym.spaces.Dict(new_obs_space)
        else:
            new_obs_space = self.unwrapped.observation_space

        self.observation_space = self.unwrapped.observation_space = new_obs_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
