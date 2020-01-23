import os
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch

from utils.utils import log, memory_consumption_mb


class TaskType(Enum):
    INIT, TERMINATE, RESET, INIT_TENSORS, ROLLOUT_STEP, POLICY_STEP, TRAIN, UPDATE_WEIGHTS, PBT, EMPTY = range(10)


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


def iter_dicts_recursively(d1, d2):
    """Assuming dicts have the exact same structure."""
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


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


def cuda_index_for_policy(policy_id):
    num_gpus = torch.cuda.device_count()
    gpu_idx = policy_id % num_gpus
    return gpu_idx


def device_for_policy(policy_id):
    num_gpus = torch.cuda.device_count()
    gpu_idx = cuda_index_for_policy(policy_id)
    device = torch.device('cuda', index=gpu_idx)
    log.debug('Using GPU #%d (total num gpus: %d) for policy %d...', gpu_idx, num_gpus, policy_id)
    return device


def cuda_envvars(policy_id):
    gpu_idx = cuda_index_for_policy(policy_id)

    cuda_envvar = 'CUDA_VISIBLE_DEVICES'
    if cuda_envvar not in os.environ:
        os.environ[cuda_envvar] = ','.join(str(g) for g in range(torch.cuda.device_count()))

    available_gpus = os.environ[cuda_envvar]
    available_gpus = [int(g) for g in available_gpus.split(',')]
    if len(available_gpus) > 0:
        gpu_to_use = available_gpus[gpu_idx]
        os.environ[cuda_envvar] = str(gpu_to_use)
        log.info('Set environment var %s to %r', cuda_envvar, os.environ[cuda_envvar])


def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
    gpu_cache_mb = torch.cuda.memory_cached(device) / 1e6
    stats = {
        f'memory_{process}': memory_mb,
        f'gpu_mem_{process}': gpu_mem_mb,
        f'gpu_cache_{process}': gpu_cache_mb,
    }
    return stats
