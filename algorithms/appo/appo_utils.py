import os
import threading
from collections import OrderedDict, deque

import gym
import numpy as np
import torch
from gym import spaces, Wrapper

from algorithms.utils.multi_agent import MultiAgentWrapper
from envs.create_env import create_env
from utils.get_available_gpus import get_available_gpus_without_triggering_pytorch_cuda_initialization
from utils.utils import log, memory_consumption_mb

CUDA_ENVVAR = 'CUDA_VISIBLE_DEVICES'


class TaskType:
    INIT, TERMINATE, RESET, ROLLOUT_STEP, POLICY_STEP, TRAIN, INIT_MODEL, PBT, EMPTY = range(9)


class DictObservationsWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.observation_space = gym.spaces.Dict(dict(obs=self.observation_space))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, done, info


def make_env_func(cfg, env_config):
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not hasattr(env, 'num_agents') or env.num_agents <= 1:
        env = MultiAgentWrapper(env)
    if not isinstance(env.observation_space, spaces.Dict):
        env = DictObservationsWrapper(env)
    return env


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


def copy_dict_structure(d):
    """Copy dictionary layout without copying the actual values (populated with Nones)."""
    d_copy = type(d)()
    _copy_dict_structure_func(d, d_copy)
    return d_copy


def _copy_dict_structure_func(d, d_copy):
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            d_copy[key] = type(value)()
            _copy_dict_structure_func(value, d_copy[key])
        else:
            d_copy[key] = None


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


def set_global_cuda_envvars():
    available_gpus = get_available_gpus_without_triggering_pytorch_cuda_initialization(os.environ)
    if CUDA_ENVVAR not in os.environ:
        os.environ[CUDA_ENVVAR] = available_gpus
    os.environ[f'{CUDA_ENVVAR}_backup_'] = os.environ[CUDA_ENVVAR]
    os.environ[CUDA_ENVVAR] = ''


def cuda_envvars(policy_id):
    orig_visible_devices = os.environ[f'{CUDA_ENVVAR}_backup_']
    available_gpus = [int(g) for g in orig_visible_devices.split(',')]
    log.info('Available GPUs: %r', available_gpus)

    # it is crucial to proper CUDA_VISIBLE_DEVICES properly before calling any torch.cuda methods, e.g. device_count()
    # this is why we're forced to use the env vars

    num_gpus = len(available_gpus)
    if num_gpus == 0:
        raise RuntimeError('This app requires a GPU and none seem to be available, sorry')

    gpu_idx_to_use = available_gpus[policy_id % num_gpus]
    os.environ[CUDA_ENVVAR] = str(gpu_idx_to_use)
    log.info('Set environment var %s to %r for policy %d', CUDA_ENVVAR, os.environ[CUDA_ENVVAR], policy_id)

    log.debug('Visible devices: %r', torch.cuda.device_count())


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


def tensor_batch_size(tensor_batch):
    for _, _, v in iterate_recursively(tensor_batch):
        return v.shape[0]


class TensorBatcher:
    def __init__(self, batch_pool):
        self.batch_pool = batch_pool

    def cat(self, dict_of_tensor_arrays, macro_batch_size, timing):
        """
        Here 'macro_batch' is the overall size of experience per iteration.
        Macro-batch = mini-batch * num_batches_per_iteration
        """

        tensor_batch = self.batch_pool.get()

        if tensor_batch is not None:
            old_batch_size = tensor_batch_size(tensor_batch)
            if old_batch_size != macro_batch_size:
                # this can happen due to PBT changing batch size during the experiment
                log.warning('Tensor macro-batch size changed from %d to %d!', old_batch_size, macro_batch_size)
                log.warning('Discarding the cached tensor batch!')
                del tensor_batch
                tensor_batch = None

        if tensor_batch is None:
            tensor_batch = copy_dict_structure(dict_of_tensor_arrays)
            log.info('Allocating new CPU tensor batch (could not get from the pool)')

            for d1, cache_d, key, tensor_arr, _ in iter_dicts_recursively(dict_of_tensor_arrays, tensor_batch):
                cache_d[key] = torch.cat(tensor_arr, dim=0).pin_memory()

        else:
            with timing.add_time('batcher_mem'):
                for d1, cache_d, key, tensor_arr, cache_t in iter_dicts_recursively(dict_of_tensor_arrays, tensor_batch):
                    offset = 0
                    for t in tensor_arr:
                        first_dim = t.shape[0]
                        cache_t[offset:offset + first_dim].copy_(t)
                        offset += first_dim

        return tensor_batch


class ObjectPool:
    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        self.pool = deque([], maxlen=self.pool_size)
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if len(self.pool) <= 0:
                return None

            obj = self.pool.pop()
            return obj

    def put(self, obj):
        with self.lock:
            self.pool.append(obj)

    def clear(self):
        with self.lock:
            self.pool = deque([], maxlen=self.pool_size)
