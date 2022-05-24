from typing import Optional

import torch

from sample_factory.utils.utils import AttrDict


def init_torch_runtime(cfg: AttrDict, max_num_threads: Optional[int] = 1):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if max_num_threads is not None:
        torch.set_num_threads(max_num_threads)
    if cfg.device == 'gpu':
        torch.backends.cudnn.benchmark = True


def inference_context(is_serial):
    if is_serial:
        # in serial mode we use the same tensors on sampler and learner
        return torch.no_grad()
    else:
        return torch.inference_mode()
