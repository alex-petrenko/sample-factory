import copy
import os
from collections import OrderedDict

import gym
import numpy as np
import torch
from gym import spaces, Wrapper

from algorithms.utils.multi_agent import MultiAgentWrapper
from envs.create_env import create_env
from utils.utils import log, memory_consumption_mb

CUDA_ENVVAR = 'CUDA_VISIBLE_DEVICES'


class TaskType:
    INIT, TERMINATE, RESET, ROLLOUT_STEP, POLICY_STEP, TRAIN, UPDATE_WEIGHTS, PBT, EMPTY = range(9)


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


def cores_for_worker_process(worker_idx, num_workers, cpu_count):
    worker_idx_modulo = worker_idx % cpu_count

    # trying to optimally assign workers to CPU cores to minimize context switching
    # logic here is best illustrated with an example
    # 20 cores, 44 workers (why? I don't know, someone wanted 44 workers)
    # first 40 are assigned to a single core each, remaining 4 get 5 cores each

    cores = None
    whole_workers_per_core = num_workers // cpu_count
    if worker_idx < whole_workers_per_core * cpu_count:
        # these workers get an private core each
        cores = [worker_idx_modulo]
    else:
        # we're dealing with some number of workers that is less than # of cpu cores
        remaining_workers = num_workers % cpu_count
        if cpu_count % remaining_workers == 0:
            cores_to_use = cpu_count // remaining_workers
            cores = list(range(worker_idx_modulo * cores_to_use, (worker_idx_modulo + 1) * cores_to_use, 1))

    return cores
