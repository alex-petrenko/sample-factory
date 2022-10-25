import math

import gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Independent, Normal
from torch.nn import functional

from sample_factory.utils.typing import ActionSpace
from sample_factory.utils.utils import log


def calc_num_actions(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return 1
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum([calc_num_actions(a) for a in action_space])
    elif isinstance(action_space, gym.spaces.Box):
        if len(action_space.shape) != 1:
            raise Exception("Non-trivial shape Box action spaces not currently supported. Try to flatten the space.")

        return action_space.shape[0]
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported!")


def calc_num_action_parameters(action_space: ActionSpace) -> int:
    """Returns the number of paramaters required to represent the given action space."""
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum([calc_num_action_parameters(a) for a in action_space])
    elif isinstance(action_space, gym.spaces.Box):
        # one mean and one standard deviation for every action
        return np.prod(action_space.shape) * 2
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported!")


def is_continuous_action_space(action_space: ActionSpace) -> bool:
    return isinstance(action_space, gym.spaces.Box)


def get_action_distribution(action_space, raw_logits):
    """
    Create the distribution object based on provided action space and unprocessed logits.
    :param action_space: Gym action space object
    :param raw_logits: this function expects unprocessed raw logits (not after log-softmax!)
    :return: action distribution that you can sample from
    """
    assert calc_num_action_parameters(action_space) == raw_logits.shape[-1]

    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalActionDistribution(raw_logits)
    elif isinstance(action_space, gym.spaces.Tuple):
        return TupleActionDistribution(action_space, logits_flat=raw_logits)
    elif isinstance(action_space, gym.spaces.Box):
        return ContinuousActionDistribution(params=raw_logits)
    else:
        raise NotImplementedError(f"Action space type {type(action_space)} not supported!")


def sample_actions_log_probs(distribution):
    if isinstance(distribution, TupleActionDistribution):
        return distribution.sample_actions_log_probs()
    else:
        actions = distribution.sample()
        log_prob_actions = distribution.log_prob(actions)
        return actions, log_prob_actions


def argmax_actions(distribution):
    if isinstance(distribution, TupleActionDistribution):
        return distribution.argmax()
    elif hasattr(distribution, "probs"):
        return torch.argmax(distribution.probs, dim=-1)
    elif hasattr(distribution, "means"):
        return distribution.means
    else:
        raise NotImplementedError(f"Action distribution type {type(distribution)} does not support argmax!")


# noinspection PyAbstractClass
class CategoricalActionDistribution:
    def __init__(self, raw_logits):
        """
        Ctor.
        :param raw_logits: unprocessed logits, typically an output of a fully-connected layer
        """

        self.raw_logits = raw_logits
        self.log_p = self.p = None

    @property
    def probs(self):
        if self.p is None:
            self.p = functional.softmax(self.raw_logits, dim=-1)
        return self.p

    @property
    def log_probs(self):
        if self.log_p is None:
            self.log_p = functional.log_softmax(self.raw_logits, dim=-1)
        return self.log_p

    def sample_gumbel(self):
        sample = torch.argmax(self.raw_logits - torch.empty_like(self.raw_logits).exponential_().log_(), -1)
        return sample

    def sample(self):
        samples = torch.multinomial(self.probs, 1, True)
        return samples

    def log_prob(self, value):
        value = value.long()
        log_probs = torch.gather(self.log_probs, -1, value).view(-1)
        return log_probs

    def entropy(self):
        p_log_p = self.log_probs * self.probs
        return -p_log_p.sum(-1)

    def _kl(self, other_log_probs):
        probs, log_probs = self.probs, self.log_probs
        kl = probs * (log_probs - other_log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_inverse(self, other_log_probs):
        kl = torch.exp(other_log_probs) * (other_log_probs - self.log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_symmetric(self, other_log_probs):
        return 0.5 * (self._kl(other_log_probs) + self._kl_inverse(other_log_probs))

    def symmetric_kl_with_uniform_prior(self):
        probs, log_probs = self.probs, self.log_probs
        num_categories = log_probs.shape[-1]
        uniform_prob = 1 / num_categories
        log_uniform_prob = math.log(uniform_prob)

        return 0.5 * (
            (probs * (log_probs - log_uniform_prob)).sum(dim=-1)
            + (uniform_prob * (log_uniform_prob - log_probs)).sum(dim=-1)
        )

    def kl_divergence(self, other):
        return self._kl(other.log_probs)

    def dbg_print(self):
        dbg_info = dict(
            entropy=self.entropy().mean(),
            min_logit=self.raw_logits.min(),
            max_logit=self.raw_logits.max(),
            min_prob=self.probs.min(),
            max_prob=self.probs.max(),
        )

        msg = ""
        for key, value in dbg_info.items():
            msg += f"{key}={value.cpu().item():.3f} "
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
        self.logit_lengths = [calc_num_action_parameters(s) for s in action_space.spaces]
        self.split_logits = torch.split(logits_flat, self.logit_lengths, dim=1)
        self.action_lengths = [calc_num_actions(s) for s in action_space.spaces]

        assert len(self.split_logits) == len(action_space.spaces)

        self.distributions = []
        for i, space in enumerate(action_space.spaces):
            self.distributions.append(get_action_distribution(space, self.split_logits[i]))

    @staticmethod
    def _flatten_actions(list_of_action_batches):
        batch_of_action_tuples = torch.cat(list_of_action_batches, 1)
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

    def argmax(self):
        list_of_action_batches = [argmax_actions(d) for d in self.distributions]
        return torch.cat(list_of_action_batches).unsqueeze(0)

    def log_prob(self, actions):
        # split into batches of actions from individual distributions
        list_of_action_batches = torch.split(actions, self.action_lengths, dim=1)

        log_probs = self._calc_log_probs(list_of_action_batches)
        return log_probs

    def entropy(self):
        entropies = [d.entropy().unsqueeze(dim=1) for d in self.distributions]
        entropies = torch.cat(entropies, dim=1)
        entropy = entropies.sum(dim=1)
        return entropy

    def kl_divergence(self, other):
        kls = [d.kl_divergence(other_d).unsqueeze(dim=1) for d, other_d in zip(self.distributions, other.distributions)]

        kls = torch.cat(kls, dim=1)
        kl = kls.sum(dim=1)
        return kl

    def symmetric_kl_with_uniform_prior(self):
        sym_kls = [d.symmetric_kl_with_uniform_prior().unsqueeze(dim=1) for d in self.distributions]
        sym_kls = torch.cat(sym_kls, dim=1)
        sym_kl = sym_kls.sum(dim=1)
        return sym_kl

    def dbg_print(self):
        for d in self.distributions:
            d.dbg_print()


# noinspection PyAbstractClass
class ContinuousActionDistribution(Independent):
    stddev_min: float = 1e-4
    stddev_max: float = 1e4

    def __init__(self, params):
        self.means, self.log_std, self.stddevs = self._init_impl(params, self.stddev_min, self.stddev_max)
        normal_dist = Normal(self.means, self.stddevs, validate_args=False)
        super().__init__(normal_dist, 1, validate_args=False)

    @staticmethod
    @torch.jit.script
    def _init_impl(params: Tensor, stddev_min: float, stddev_max: float):
        # using torch.chunk here is slightly faster than plain indexing
        means, log_std = torch.chunk(params, 2, dim=1)
        stddevs = log_std.exp()
        stddevs = torch.clamp(stddevs, stddev_min, stddev_max)
        return means, log_std, stddevs

    def kl_divergence(self, other):
        kl = torch.distributions.kl.kl_divergence(self, other)
        return kl

    def summaries(self):
        return dict(
            action_mean=self.means.mean(),
            action_mean_min=self.means.min(),
            action_mean_max=self.means.max(),
            action_log_std_mean=self.log_std.mean(),
            action_log_std_min=self.log_std.min(),
            action_log_std_max=self.log_std.max(),
            action_stddev_mean=self.stddev.mean(),
            action_stddev_min=self.stddev.min(),
            action_stddev_max=self.stddev.max(),
        )
