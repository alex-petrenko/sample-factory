from collections import namedtuple

import gym
import torch
from torch.distributions import Categorical


def calc_num_logits(action_space):
    """Returns the number of logits required to represent the given action space."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum(space.n for space in action_space.spaces)
    else:
        raise NotImplementedError(f'Action space type {type(action_space)} not supported!')


def get_action_distribution(action_space, logits):
    assert calc_num_logits(action_space) == logits.shape[-1]

    if isinstance(action_space, gym.spaces.Discrete):
        return Categorical(logits=logits)
    elif isinstance(action_space, gym.spaces.Tuple):
        return TupleActionDistribution(action_space, logits_flat=logits)


# TupleActions = namedtuple('TupleActions', ['action_batches'])


class TupleActionDistribution:
    """
    Basically, a tuple of independent action distributions.
    Useful when the environment requires multiple independent action heads, e.g.:
     - moving in the environment
     - selecting a weapon
     - jumping
     - strafing

    Empirically, it seems to be better to represent such an action distribution as a tuple of independent action
    distributions, rather than a one-hot over potentially big cartesian product of all action spaces, like it's
    usually done in Atari.

    Entropy of such a distribution is just a sum of entropies of individual distributions.

    """

    def __init__(self, action_space, logits_flat):
        self.logit_lengths = [calc_num_logits(s) for s in action_space.spaces]
        self.split_logits = torch.split(logits_flat, self.logit_lengths, dim=1)
        assert len(self.split_logits) == len(action_space.spaces)

        self.distributions = [get_action_distribution(s, l) for s, l in zip(action_space.spaces, self.split_logits)]

    def sample(self):
        list_of_action_batches = [d.sample() for d in self.distributions]
        batch_of_lists_of_actions = torch.stack(list_of_action_batches).transpose(0, 1)
        return batch_of_lists_of_actions

    def log_prob(self, actions):
        num_distributions = len(self.distributions)

        # split into batches of actions from individual distributions
        actions_split = torch.chunk(actions, num_distributions, dim=1)

        # calculate batched log probs for every distribution
        log_probs = [d.log_prob(a.squeeze(dim=1)) for d, a in zip(self.distributions, actions_split)]
        log_probs = [lp.unsqueeze(dim=1) for lp in log_probs]

        # concatenate and calculate sum of individual log-probs
        # this is valid under the assumption that action distributions are independent
        log_probs = torch.cat(log_probs, dim=1)
        log_probs = log_probs.sum(dim=1)
        return log_probs

    def entropy(self):
        entropies = [d.entropy().unsqueeze(1) for d in self.distributions]
        entropies = torch.cat(entropies, dim=1)
        entropy = entropies.sum(dim=1)
        return entropy
