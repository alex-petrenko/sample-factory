from typing import Optional

import numpy as np
import torch

from sample_factory.utils.utils import AttrDict


def init_torch_runtime(cfg: AttrDict, max_num_threads: Optional[int] = 1):
    torch.multiprocessing.set_sharing_strategy("file_system")
    if max_num_threads is not None:
        torch.set_num_threads(max_num_threads)
    if cfg.device == "gpu":
        torch.backends.cudnn.benchmark = True


def inference_context(is_serial):
    if is_serial:
        # in serial mode we use the same tensors on sampler and learner
        return torch.no_grad()
    else:
        return torch.inference_mode()


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value
