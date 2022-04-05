import sys
from typing import Final, Dict

import torch
from gym import spaces
from torch import nn, Tensor

from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace, RunningMeanStdDictInPlace, \
    running_mean_std_summaries
from sample_factory.algorithms.appo.appo_utils import copy_dict_structure, iter_dicts_recursively
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.const: Final = 42.0

        space = spaces.Box(-100, 100, shape=(60, ))
        obs_space = spaces.Dict(spaces=dict(obs=space))
        self.rm = RunningMeanStdDictInPlace(obs_space)
        # self.rm = torch.jit.script(self.rm)

    @staticmethod
    def _clone_tensordict(obs_dict: Dict[str, Tensor]):
        obs_clone = copy_dict_structure(obs_dict)  # creates an identical dict but with empty values
        for d, d_clone, k, x, _ in iter_dicts_recursively(obs_dict, obs_clone):
            if x.dtype != torch.float:
                d_clone[k] = x.float()  # this will create a copy of a tensor
            else:
                d_clone[k] = x.clone()  # otherwise, we explicitly clone it

        return obs_clone

    def forward(self, x):
        y = self._clone_tensordict(x)
        y['obs'] = y['obs'] * self.const
        self.rm(y)
        return y

    def summaries(self):
        return running_mean_std_summaries(self.rm)


def main():
    device = torch.device('cuda', index=0)

    m = MyModule().to(device)

    t = dict(obs=torch.randn([32768, 60]).to(device))
    log.debug('%r', m(t)['obs'][0])

    timing = Timing()

    res = torch.zeros([1]).to(device)

    log.debug('starting...')
    with timing.timeit('norm'):
        for i in range(10000):
            normalized = m(t)
            res += normalized['obs'].mean()

    log.debug(res)
    log.debug(timing)
    log.debug(m.summaries())


if __name__ == '__main__':
    sys.exit(main())
