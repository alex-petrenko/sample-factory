import math

import gym
import numpy as np
import torch
from torch.distributions import Categorical, Normal, Independent
from torch.nn import functional

from utils.utils import log


def calc_num_actions(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return 1
    elif isinstance(action_space, gym.spaces.Tuple):
        return len(action_space.spaces)
    elif isinstance(action_space, gym.spaces.Box):
        if len(action_space.shape) != 1:
            raise Exception('Non-trivial shape Box action spaces not currently supported. Try to flatten it.')

        return action_space.shape[0]
    else:
        raise NotImplementedError(f'Action space type {type(action_space)} not supported!')


def calc_num_logits(action_space):
    """Returns the number of logits required to represent the given action space."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum(space.n for space in action_space.spaces)
    elif isinstance(action_space, gym.spaces.Box):
        # regress one mean and one standard deviation for every action
        return np.prod(action_space.shape) * 2
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

    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalActionDistribution(functional.log_softmax(raw_logits, dim=1))
    elif isinstance(action_space, gym.spaces.Tuple):
        return TupleActionDistribution(action_space, logits_flat=raw_logits)
    elif isinstance(action_space, gym.spaces.Box):
        return ContinuousActionDistribution(params=raw_logits)
    else:
        raise NotImplementedError(f'Action space type {type(action_space)} not supported!')


def sample_actions_log_probs(distribution):
    if isinstance(distribution, TupleActionDistribution):
        return distribution.sample_actions_log_probs()
    else:
        actions = distribution.sample()
        log_prob_actions = distribution.log_prob(actions)
        return actions, log_prob_actions


# noinspection PyAbstractClass
class CategoricalActionDistribution(Categorical):
    """
    A thin wrapper on top of standard PyTorch categorical, with some functionality added.

    """

    def __init__(self, raw_logits):
        """
        Ctor.
        :param raw_logits: unprocessed logits, typically an output of a fully-connected layer
        """
        log_probabilities = functional.log_softmax(raw_logits, dim=1)
        super().__init__(logits=log_probabilities)

    def _kl(self, other_log_probs):
        probs, log_probs = self.probs, self.logits
        kl = probs * (log_probs - other_log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_inverse(self, other_log_probs):
        probs, log_probs = self.probs, self.logits
        kl = torch.exp(other_log_probs) * (other_log_probs - log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_symmetric(self, other_log_probs):
        return 0.5 * (self._kl(other_log_probs) + self._kl_inverse(other_log_probs))

    def kl_divergence(self, other):
        return self._kl(other.logits)

    def dbg_print(self):
        dbg_info = dict(
            entropy=self.entropy().mean(),
            min_logit=self.logits.min(),
            max_logit=self.logits.max(),
            min_prob=self.probs.min(),
            max_prob=self.probs.max(),
        )

        msg = ''
        for key, value in dbg_info.items():
            msg += f'{key}={value.cpu().item():.3f} '
        log.debug(msg)


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

        self.distributions = []
        for i, space in enumerate(action_space.spaces):
            self.distributions.append(get_action_distribution(space, self.split_logits[i]))

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

    def kl_divergence(self, other):
        kls = [
            d.kl_divergence(other_d).unsqueeze(dim=1)
            for d, other_d
            in zip(self.distributions, other.distributions)
        ]

        kls = torch.cat(kls, dim=1)
        kl = kls.sum(dim=1)
        return kl

    def dbg_print(self):
        for d in self.distributions:
            d.dbg_print()


# noinspection PyAbstractClass
class ContinuousActionDistribution(Independent):
    dist_min_clamp = math.log(1e-6)
    dist_max_clamp = math.log(1e6)

    def __init__(self, params):
        num_actions = params.shape[-1] // 2
        means, log_std = torch.split(params, num_actions, dim=1)
        log_std = log_std.clamp(self.dist_min_clamp, self.dist_max_clamp)
        stddev = log_std.exp()

        normal_dist = Normal(means, stddev)
        super().__init__(normal_dist, 1)

    def kl_divergence(self, other):
        kl = torch.distributions.kl.kl_divergence(self, other)
        return kl
