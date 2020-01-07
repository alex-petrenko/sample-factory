from collections import OrderedDict
from enum import Enum

import numpy as np
import torch

from utils.utils import log


class TaskType(Enum):
    INIT, TERMINATE, RESET, INIT_TENSORS, ROLLOUT_STEP, POLICY_STEP, TRAIN, UPDATE_WEIGHTS, EMPTY = range(9)


def set_step_data(dictionary, key, data):
    if isinstance(data, np.ndarray):
        torch_data = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        torch_data = data
    elif isinstance(data, (int, float, bool, list, tuple, np.float32)):
        torch_data = torch.tensor(data)
    else:
        raise RuntimeError('Unsupported data type!')

    if key not in dictionary:
        dictionary[key] = torch_data.cpu().clone().detach()  # this is slow, but we do it only once
        dictionary[key].share_memory_()
    else:
        dictionary[key].copy_(torch_data)


def dict_of_lists_append(dict_of_lists, new_data):
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(x)
        else:
            dict_of_lists[key] = [x]


def iterate_recursively(d):
    """
    Generator for a dictionary that can potentially include other dictionaries.
    Yields tuples of (dict, key, value), where key, value are "leaf" elements of the "dict".

    """
    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = dict()

    for d in list_of_dicts:
        for key, x in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []

            dict_of_lists[key].append(x)

    return dict_of_lists


def extend_array_by(x, extra_len):
    """Assuming the array is currently not empty."""
    if extra_len <= 0:
        return x

    last_elem = x[-1]
    tail = [last_elem] * extra_len
    tail = np.stack(tail)
    return np.append(x, tail, axis=0)


def device_for_policy(policy_id):
    num_gpus = torch.cuda.device_count()
    gpu_idx = policy_id % num_gpus
    device = torch.device('cuda', index=gpu_idx)
    log.debug('Using GPU #%d (total num gpus: %d) for policy %d...', gpu_idx, num_gpus, policy_id)
    return device
