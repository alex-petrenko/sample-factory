import random

import numpy as np


class PolicyManager:
    """
    This class currently implements the most simple mapping between agents in the envs and their associated policies.
    We just pick a random policy from the population for every agent at the beginning of the episode.

    Methods of this class can potentially be overloaded to provide a more clever mapping, e.g. we can minimize the
    number of different policies per rollout worker thus minimizing the amount of communication required.
    """

    def __init__(self, num_agents, num_policies):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = num_agents
        self.num_policies = num_policies

    def get_policy_for_agent(self, unused_agent_idx):
        return self.rng.randint(0, self.num_policies)

