import random

import numpy as np
import time


class PolicyManager:
    """
    Helper class responsible for randomly picking the agent-policy mapping during training.
    In population-based training scenarios we can gain performance by minimizing the number of policies that control
    agents on a specific policy worker.
    Consider a 2-agent environment and a 12-policy population-based training config. On each worker we will have
    multiple copies of this environment, and a default regime would be to randomly sample a policy index for every
    agent in the beginning of the episode, which will likely lead to all 12 policies being involved with the
    rollout worker. E.g. at the end of each simulation step we will have to send messages to all 12 policy workers,
    and then wait for all of them to finish before the next simulation step can be started.

    An alternative strategy is to randomly pick only two policies per worker, this will significanly reduce the amount
    of communication. We can re-sample these two policies every N seconds to ensure that all pairs of policies
    interact equally often. Also the fact that we're running many workers will ensure that the entire population is
    trained uniformly.

    """

    def __init__(self, num_agents, num_policies):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = num_agents
        self.num_policies = num_policies

        self.curr_policies_per_agent = None
        self._resample()

        self.last_sampled = time.time()
        self.resample_interval_sec = 10

    def _resample(self):
        self.curr_policies_per_agent = [self.rng.randint(0, self.num_policies) for _ in range(self.num_agents)]

    def get_policy_for_agent(self, agent_idx):
        now = time.time()
        if now - self.last_sampled > self.resample_interval_sec:
            self._resample()
            self.last_sampled = now

        return self.curr_policies_per_agent[agent_idx]

