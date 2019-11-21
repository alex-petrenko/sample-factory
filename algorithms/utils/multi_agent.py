from gym import Wrapper, spaces


class MultiAgentWrapper(Wrapper):
    """
    This wrapper allows us to treat a single-agent environment as multi-agent with 1 agent.
    That is, the data (obs, rewards, etc.) is converted into lists of length 1

    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents = 1

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return [obs]

    def step(self, action):
        action = action[0]
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()
        return [obs], [rew], [done], [info]
