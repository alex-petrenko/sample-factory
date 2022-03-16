"""
PyTorch module that keeps track of tensor statistics and uses it to normalize data.
All credit goes to https://github.com/Denys88/rl_games (only slightly changed here)
Thanks a lot, great module!
"""
import gym
import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.utils.utils import log


# noinspection PyAttributeOutsideInit
class RunningMeanStd(nn.Module):
    def __init__(self, input_shape, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        log.debug('RunningMeanStd input shape: %r', input_shape)
        self.input_shape = input_shape
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel

        if per_channel:
            if len(self.input_shape) == 3:
                self.axis = [0, 2, 3]
            if len(self.input_shape) == 2:
                self.axis = [0, 2]
            if len(self.input_shape) == 1:
                self.axis = [0]
            shape = self.input_shape[0]
        else:
            self.axis = [0]
            shape = input_shape

        self.register_buffer('running_mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(shape, dtype=torch.float64))
        self.register_buffer('count', torch.ones((), dtype=torch.float64))

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        return new_mean, new_var, tot_count

    def forward(self, x: Tensor):
        if self.training:
            mean = x.mean(self.axis)  # along channel axis
            var = x.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, x.size()[0],
            )

        # change shape
        if self.per_channel:
            if len(self.input_shape) == 3:
                current_mean = self.running_mean.view([1, self.input_shape[0], 1, 1]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0], 1, 1]).expand_as(x)
            elif len(self.input_shape) == 2:
                current_mean = self.running_mean.view([1, self.input_shape[0], 1]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0], 1]).expand_as(x)
            elif len(self.input_shape) == 1:
                current_mean = self.running_mean.view([1, self.input_shape[0]]).expand_as(x)
                current_var = self.running_var.view([1, self.input_shape[0]]).expand_as(x)
            else:
                raise RuntimeError(f'RunningMeanStd input shape {self.input_shape} not supported')
        else:
            current_mean = self.running_mean
            current_var = self.running_var

        if self.norm_only:
            y = x / torch.sqrt(current_var.float() + self.epsilon)
        else:
            y = x - current_mean.float()
            y.mul_(1.0 / torch.sqrt(current_var.float() + self.epsilon))
            y.clamp_(-5.0, 5.0)

        return y


class RunningMeanStdDict(nn.Module):
    def __init__(self, obs_space: gym.spaces.Dict, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStdDict, self).__init__()
        self.obs_space = obs_space
        self.running_mean_std = nn.ModuleDict({
            k: RunningMeanStd(space.shape, epsilon, per_channel, norm_only) for k, space in obs_space.spaces.items()
        })

    def forward(self, x):
        return {k: self.running_mean_std[k](v) for k, v in x.items()}

    def summaries(self):
        res = dict()
        for k in self.obs_space:
            stats = self.running_mean_std[k]
            res.update({
                f'{k}_mean': stats.running_mean.mean(),
                f'{k}_std': stats.running_var.mean(),
            })
        return res
