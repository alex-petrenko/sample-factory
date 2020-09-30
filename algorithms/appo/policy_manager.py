import random

import numpy as np


class PolicyManager:
    """
    This class currently implements the most simple mapping between agents in the envs and their associated policies.
    We just pick a random policy from the population for every agent at the beginning of the episode.

    Methods of this class can potentially be overloaded to provide a more clever mapping, e.g. we can minimize the
    number of different policies per rollout worker thus minimizing the amount of communication required.
    """

    def __init__(self, cfg, num_agents):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = num_agents
        self.num_policies = cfg.num_policies
        self.mix_policies_in_one_env = cfg.pbt_mix_policies_in_one_env

        self.resample_env_policy_every = 10  # episodes
        self.env_policies = dict()
        self.env_policy_requests = dict()

    def get_policy_for_agent(self, agent_idx, env_idx):
        num_requests = self.env_policy_requests.get(env_idx, 0)
        if num_requests % (self.num_agents * self.resample_env_policy_every) == 0:
            if self.mix_policies_in_one_env:
                self.env_policies[env_idx] = [self._sample_policy() for _ in range(self.num_agents)]
            else:
                policy = self._sample_policy()
                self.env_policies[env_idx] = [policy] * self.num_agents

        self.env_policy_requests[env_idx] = num_requests + 1
        return self.env_policies[env_idx][agent_idx]

    def _sample_policy(self):
        return self.rng.randint(0, self.num_policies)
