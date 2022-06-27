import pytest
import gym
import numpy as np
import torch
from sample_factory.algo.utils.action_distributions import calc_num_actions, calc_num_logits, get_action_distribution





class DummyTupleEnv(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        action_spaces=[],
        action_sizes=[],
    ):
        super().__init__()

        _action_spaces = []

        for space, size in zip(action_spaces, action_sizes):
            if space is gym.spaces.Discrete:
                _action_spaces.append(space(size))
            else:
                _action_spaces.append(gym.spaces.Box(low=-1, high=1, shape=(size,), dtype=np.float32))

        self.action_space = gym.spaces.Tuple(_action_spaces)

        self.observation_space = gym.spaces.Dict(
            {
                # Vector obs
                "vec": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }
        )


    def seed(self, seed=None):
        if seed is not None:
            self.observation_space.seed(seed)

    def step(self, action):
        reward = 0.0
        done = False
        return self.observation_space.sample(), reward, done, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.zeros((len(achieved_goal),))

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode="human"):
        pass


@pytest.mark.parametrize("spaces", [
            [gym.spaces.Discrete, gym.spaces.Discrete],
            [gym.spaces.Discrete, gym.spaces.Box],
            [gym.spaces.Box, gym.spaces.Box]
        ])
@pytest.mark.parametrize("sizes", [[1,1],[2,1], [1,2], [2,3]])
def test_tuple_action_distribution(spaces, sizes):
     # I like to use prime numbers for tests as it can flag problems hidden by automatic broadcasting etc
    BATCH_SIZE = 31

    assert len(spaces) > 0
    assert len(spaces) == len(sizes)

    num_actions = 0
    num_logits = 0
    num_distributions = len(spaces)
    _action_spaces = []
    for space, size in zip(spaces, sizes):
        if space is gym.spaces.Discrete:
            _action_spaces.append(space(size))
            num_actions += 1
            num_logits += size
            
        else:
            _action_spaces.append(gym.spaces.Box(low=-1, high=1, shape=(size,), dtype=np.float32))
            num_actions += size
            num_logits += size * 2

    action_space = gym.spaces.Tuple(_action_spaces)

    assert calc_num_actions(action_space) == num_actions
    assert calc_num_logits(action_space) == num_logits

    logits = torch.randn(BATCH_SIZE, num_logits)
    action_dist = get_action_distribution(action_space, logits)

    actions = action_dist.sample()
    action_log_probs = action_dist.log_prob(actions)
    entropy = action_dist.entropy()
    assert actions.size() == (BATCH_SIZE, num_actions)
    assert action_log_probs.size() == (BATCH_SIZE, )
    assert entropy.size() == (BATCH_SIZE, )

    actions, action_log_probs = action_dist.sample_actions_log_probs()
    assert actions.size() == (BATCH_SIZE, num_actions)
    assert action_log_probs.size() == (BATCH_SIZE, )
