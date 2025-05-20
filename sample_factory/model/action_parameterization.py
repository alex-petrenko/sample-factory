import math

import torch
from torch import Tensor, nn

from sample_factory.algo.utils.action_distributions import (
    calc_num_action_parameters,
    get_action_distribution,
    is_continuous_action_space,
)


class ActionsParameterization(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space


class ActionParameterizationDefault(ActionsParameterization):
    """
    A single fully-connected layer to output all parameters of the action distribution. Suitable for
    categorical action distributions, as well as continuous actions with learned state-dependent stddev.

    """

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        num_action_outputs = calc_num_action_parameters(action_space)
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)

    def forward(self, actor_core_output):
        """Just forward the FC layer and generate the distribution object."""
        action_distribution_params = self.distribution_linear(actor_core_output)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution


class ActionParameterizationContinuousNonAdaptiveStddev(ActionsParameterization):
    """Use a single learned parameter for action stddevs."""

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        assert not cfg.adaptive_stddev
        assert is_continuous_action_space(
            self.action_space
        ), "Non-adaptive stddev makes sense only for continuous action spaces"

        num_action_outputs = calc_num_action_parameters(action_space)

        # calculate only action means using the policy neural network
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs // 2)
        self.tanh_scale: float = cfg.continuous_tanh_scale
        # stddev is a single learned parameter
        initial_stddev = torch.empty([num_action_outputs // 2])
        initial_stddev.fill_(math.log(self.cfg.initial_stddev))
        self.learned_stddev = nn.Parameter(initial_stddev, requires_grad=True)

    def forward(self, actor_core_output: Tensor):
        action_means = self.distribution_linear(actor_core_output)
        if self.tanh_scale > 0:
            # scale the action means to be in the range [-tanh_scale, tanh_scale]
            # TODO: implement this for adaptive stddev case also?
            action_means = torch.tanh(action_means / self.tanh_scale) * self.tanh_scale

        batch_size = action_means.shape[0]
        action_stddevs = self.learned_stddev.repeat(batch_size, 1)
        action_distribution_params = torch.cat((action_means, action_stddevs), dim=1)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution
