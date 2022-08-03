import random

import numpy as np

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import total_num_envs
from sample_factory.utils.typing import Config


class AgentPolicyMapping:
    """
    This class currently implements the most simple mapping between agents in the envs and their associated policies.
    We just pick a random policy from the population for every agent at the beginning of the episode.

    Methods of this class can potentially be overloaded to provide a more clever mapping, e.g. we can minimize the
    number of different policies per rollout worker thus minimizing the amount of communication required.

    TODO: would be nice to have a mechanism to provide custom mappings through API.
    """

    def __init__(self, cfg: Config, env_info: EnvInfo):
        self.rng = np.random.RandomState(seed=random.randint(0, 2**32 - 1))

        self.num_agents = env_info.num_agents
        self.num_policies = cfg.num_policies
        self.mix_policies_in_one_env = (
            cfg.pbt_mix_policies_in_one_env if hasattr(cfg, "pbt_mix_policies_in_one_env") else False
        )  # TODO

        self.resample_env_policy_every = 10  # episodes
        self.env_policies = dict()
        self.env_policy_requests = dict()

        total_envs = total_num_envs(cfg)
        self.sync_mode = not cfg.async_rl
        if self.sync_mode:
            assert total_envs % self.num_policies == 0, f"{total_envs=} must be divisible by {self.num_policies=}"

    def get_policy_for_agent(self, agent_idx: int, env_idx: int, global_env_idx: int) -> int:
        if self.sync_mode:
            # env_id here is a global index of the policy
            # deterministic mapping ensures we always collect the same amount of experience per policy per iteration
            # Sync mode is an experimental feature. This code can be further improved to allow more sophisticated
            # agent-policy mapping.
            return global_env_idx % self.num_policies

        num_requests = self.env_policy_requests.get(env_idx, 0)

        # extra cheeky flag to make sure this code executes early in the training so we spot any potential problems
        early_in_the_training = num_requests < 5
        if num_requests % (self.num_agents * self.resample_env_policy_every) == 0 or early_in_the_training:
            if self.mix_policies_in_one_env:
                self.env_policies[env_idx] = [self._sample_policy() for _ in range(self.num_agents)]
            else:
                policy = self._sample_policy()
                self.env_policies[env_idx] = [policy] * self.num_agents

        self.env_policy_requests[env_idx] = num_requests + 1
        return self.env_policies[env_idx][agent_idx]

    def _sample_policy(self):
        return self.rng.randint(0, self.num_policies)
