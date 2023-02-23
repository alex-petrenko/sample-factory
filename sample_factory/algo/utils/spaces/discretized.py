from gymnasium.spaces import Discrete


class Discretized(Discrete):
    def __init__(self, n, min_action, max_action):
        super().__init__(n)

        self.min_action = min_action
        self.max_action = max_action

    def to_continuous(self, discrete_action):
        step = (self.max_action - self.min_action) / (self.n - 1)  # min=-1, max=1, n=11, step=0.2
        action = self.min_action + discrete_action * step
        return action
