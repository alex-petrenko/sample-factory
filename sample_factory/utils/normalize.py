"""
Data normalization and scaling, used mostly for env observations.

Implemented as a PyTorch module so that any learned parameters (such as data statistics
in RunningMeanStd is saved/loaded to/from checkpoints via normal mechanisms.

We do this normalization step as preprocessing before inference or learning. It is important to do it only once
before each learning iteration (not before each epoch or minibatch), since this is just redundant work.

If no data normalization is needed we just keep the original data.
Otherwise, we create a copy of data and do all of the operations operations in-place.
"""
from typing import Dict

import torch
from torch import nn

from sample_factory.algo.utils.misc import EPS
from sample_factory.algo.utils.running_mean_std import RunningMeanStdDictInPlace, running_mean_std_summaries
from sample_factory.utils.dicts import copy_dict_structure, iter_dicts_recursively


class ObservationNormalizer(nn.Module):
    def __init__(self, obs_space, cfg):
        super().__init__()

        self.sub_mean = cfg.obs_subtract_mean
        self.scale = cfg.obs_scale

        self.running_mean_std = None
        if cfg.normalize_input:
            self.running_mean_std = RunningMeanStdDictInPlace(obs_space, cfg.normalize_input_keys)

        self.should_sub_mean = abs(self.sub_mean) > EPS
        self.should_scale = abs(self.scale - 1.0) > EPS
        self.should_normalize = self.should_sub_mean or self.should_scale or self.running_mean_std is not None

    @staticmethod
    def _clone_tensordict(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_clone = copy_dict_structure(obs_dict)  # creates an identical dict but with None values
        for d, d_clone, k, x, _ in iter_dicts_recursively(obs_dict, obs_clone):
            if x.dtype != torch.float:
                # type conversion requires a copy, do this check to make sure we don't do it twice
                d_clone[k] = x.float()  # this will create a copy of a tensor
            else:
                d_clone[k] = x.clone()  # otherwise, we explicitly clone it since normalization is in-place

        return obs_clone

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.should_normalize:
            return obs_dict

        with torch.no_grad():
            # since we are creating a clone, it is safe to use in-place operations
            obs_clone = self._clone_tensordict(obs_dict)

            # subtraction of mean and scaling is only applied to default "obs"
            # this should be modified for custom obs dicts
            if self.should_sub_mean:
                obs_clone["obs"].sub_(self.sub_mean)

            if self.should_scale:
                obs_clone["obs"].mul_(1.0 / self.scale)

            if self.running_mean_std:
                self.running_mean_std(obs_clone)  # in-place normalization

        return obs_clone

    def summaries(self) -> Dict:
        res = dict()
        if self.running_mean_std:
            res.update(running_mean_std_summaries(self.running_mean_std))
        return res
