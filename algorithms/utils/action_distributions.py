import gym
import torch
from torch.nn import functional
from torch.distributions import Categorical

from algorithms.memento.mem_wrapper import MemActionSpace, MemCategorical


def calc_num_logits(action_space):
    """Returns the number of logits required to represent the given action space."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum(space.n for space in action_space.spaces)
    else:
        raise NotImplementedError(f'Action space type {type(action_space)} not supported!')


def get_action_distribution(action_space, raw_logits):
    """
    Create the distribution object based on provided action space and unprocessed logits.
    :param action_space: Gym action space object
    :param raw_logits: this function expects unprocessed raw logits (not after log-softmax!)
    :return: action distribution that you can sample from
    """
    assert calc_num_logits(action_space) == raw_logits.shape[-1]

    if isinstance(action_space, MemActionSpace):
        log_probabilities = functional.log_softmax(raw_logits, dim=1)
        return MemCategorical(logits=log_probabilities, prior_probs=action_space.prior_probs)
    if isinstance(action_space, gym.spaces.Discrete):
        log_probabilities = functional.log_softmax(raw_logits, dim=1)
        return Categorical(logits=log_probabilities)
    elif isinstance(action_space, gym.spaces.Tuple):
        return TupleActionDistribution(action_space, logits_flat=raw_logits)
    else:
        raise NotImplementedError(f'Action space type {type(action_space)} not supported!')


def sample_actions_log_probs(distribution):
    if isinstance(distribution, TupleActionDistribution):
        return distribution.sample_actions_log_probs()
    else:
        actions = distribution.sample()
        log_prob_actions = distribution.log_prob(actions)
        return actions, log_prob_actions


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

    @staticmethod
    def _flatten_actions(list_of_action_batches):
        batch_of_action_tuples = torch.stack(list_of_action_batches).transpose(0, 1)
        return batch_of_action_tuples

    def _calc_log_probs(self, list_of_action_batches):
        # calculate batched log probs for every distribution
        log_probs = [d.log_prob(a) for d, a in zip(self.distributions, list_of_action_batches)]
        log_probs = [lp.unsqueeze(dim=1) for lp in log_probs]

        # concatenate and calculate sum of individual log-probs
        # this is valid under the assumption that action distributions are independent
        log_probs = torch.cat(log_probs, dim=1)
        log_probs = log_probs.sum(dim=1)

        return log_probs

    def sample_actions_log_probs(self):
        list_of_action_batches = [d.sample() for d in self.distributions]
        batch_of_action_tuples = self._flatten_actions(list_of_action_batches)
        log_probs = self._calc_log_probs(list_of_action_batches)
        return batch_of_action_tuples, log_probs

    def sample(self):
        list_of_action_batches = [d.sample() for d in self.distributions]
        return self._flatten_actions(list_of_action_batches)

    def log_prob(self, actions):
        # split into batches of actions from individual distributions
        list_of_action_batches = torch.chunk(actions, len(self.distributions), dim=1)
        list_of_action_batches = [a.squeeze(dim=1) for a in list_of_action_batches]

        log_probs = self._calc_log_probs(list_of_action_batches)
        return log_probs

    def entropy(self):
        entropies = [d.entropy().unsqueeze(dim=1) for d in self.distributions]
        entropies = torch.cat(entropies, dim=1)
        entropy = entropies.sum(dim=1)
        return entropy
