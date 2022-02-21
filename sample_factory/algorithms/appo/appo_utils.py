import os
import threading
from collections import OrderedDict, deque
from typing import Iterable, Callable

import gym
import numpy as np
import torch
from gym import spaces, Wrapper

from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.get_available_gpus import get_gpus_without_triggering_pytorch_cuda_initialization
from sample_factory.utils.utils import log, memory_consumption_mb

CUDA_ENVVAR = 'CUDA_VISIBLE_DEVICES'


class TaskType:
    INIT, TERMINATE, RESET, ROLLOUT_STEP, POLICY_STEP, TRAIN, INIT_MODEL, PBT, UPDATE_ENV_STEPS, EMPTY = range(10)


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
    if not is_multiagent_env(env):
        env = MultiAgentWrapper(env)
    if not isinstance(env.observation_space, spaces.Dict):
        env = DictObservationsWrapper(env)
    return env


# TODO: remove code duplication!
class DictObservationsWrapperV2(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.observation_space = gym.spaces.Dict(dict(obs=self.observation_space))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return dict(obs=obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return dict(obs=obs), rew, done, info


class TensorWrapper(Wrapper):
    """Ensures that the env returns a dictionary of tensors for observations, and tensors for rewards and dones."""

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents

        self._convert_obs_func = dict()
        self._convert_rew_func = self._convert_dones_func = None

    def _convert(self, obs):
        result = dict()
        for key, value in obs.items():
            result[key] = self._convert_obs_func[key](value)
        return result

    @staticmethod
    def _get_convert_func(x) -> Callable:
        """Depending on type of x, determines the conversion function from x to a tensor."""
        if isinstance(x, torch.Tensor):
            return lambda x_: x_  # do nothing
        elif isinstance(x, np.ndarray):
            return lambda x_: torch.from_numpy(x_)
        elif isinstance(x, (list, tuple)):
            if isinstance(x[0], np.ndarray) or isinstance(x[0], (list, tuple)):
                # creating a tensor from a list of numpy.ndarrays is extremely slow
                # so we first create a numpy array which is then converted to a tensor
                return lambda x_: torch.tensor(np.array(x_))
            elif isinstance(x[0], torch.Tensor):
                return lambda x_: torch.tensor(x_)
            else:
                # just make a tensor and hope for the best
                # leave it like this for now, we can add more cases later if we need to
                return lambda x_: torch.tensor(x_)
        else:
            raise RuntimeError(f'Cannot convert data type {type(x)} to torch.Tensor')

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        assert isinstance(obs, dict)

        for key, value in obs.items():
            if key not in self._convert_obs_func:
                self._convert_obs_func[key] = self._get_convert_func(value)

        return self._convert(obs)

    def step(self, action):
        obs, rew, dones, infos = self.env.step(action)
        obs = self._convert(obs)

        if not self._convert_rew_func:
            # the only way to reliably find out the format of data is to actually look what the environment returns
            self._convert_rew_func = self._get_convert_func(rew)
            self._convert_dones_func = self._get_convert_func(dones)

        rew = self._convert_rew_func(rew)
        dones = self._convert_dones_func(dones)
        return obs, rew, dones, infos


# TODO: remove code duplication!
def make_env_func_v2(cfg, env_config):
    """
    This should yield an environment that always returns a dict of PyTorch tensors (CPU- or GPU-side) or
    a dict of numpy arrays or a dict of lists (depending on what the environment returns in the first place).
    In case
    """
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not is_multiagent_env(env):
        env = MultiAgentWrapper(env)
    if not isinstance(env.observation_space, spaces.Dict):
        env = DictObservationsWrapperV2(env)

    # At this point we can be sure that our environment outputs a dictionary of lists (or numpy arrays or tensors)
    # containing obs, rewards, etc. for each agent in the environment. If it wasn't true to begin with, we guaranteed
    # that by adding wrappers above.
    # Now we just want the environment to return a tensor dict for observations and tensors for rewards and dones.
    # We leave infos intact for now, because format of infos isn't really specified and can be inconsistent between
    # timesteps.

    env = TensorWrapper(env)

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
    """
    Assuming structure of d1 is strictly included into d2.
    I.e. each key at each recursion level is also present in d2. This is also true when d1 and d2 have the same
    structure.
    """
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


def set_global_cuda_envvars(cfg):
    if cfg.device == 'cpu':
        available_gpus = ''
    else:
        available_gpus = get_gpus_without_triggering_pytorch_cuda_initialization(os.environ)

    if CUDA_ENVVAR not in os.environ:
        os.environ[CUDA_ENVVAR] = available_gpus
    os.environ[f'{CUDA_ENVVAR}_backup_'] = os.environ[CUDA_ENVVAR]
    os.environ[CUDA_ENVVAR] = ''


def get_available_gpus():
    orig_visible_devices = os.environ[f'{CUDA_ENVVAR}_backup_']
    available_gpus = [int(g) for g in orig_visible_devices.split(',') if g]
    return available_gpus


def set_gpus_for_process(process_idx, num_gpus_per_process, process_type, gpu_mask=None):
    available_gpus = get_available_gpus()
    if gpu_mask is not None:
        assert len(available_gpus) >= len(available_gpus)
        available_gpus = [available_gpus[g] for g in gpu_mask]
    num_gpus = len(available_gpus)
    gpus_to_use = []

    if num_gpus == 0:
        os.environ[CUDA_ENVVAR] = ''
        log.debug('Not using GPUs for %s process %d', process_type, process_idx)
    else:
        first_gpu_idx = process_idx * num_gpus_per_process
        for i in range(num_gpus_per_process):
            index_mod_num_gpus = (first_gpu_idx + i) % num_gpus
            gpus_to_use.append(available_gpus[index_mod_num_gpus])

        os.environ[CUDA_ENVVAR] = ','.join([str(g) for g in gpus_to_use])
        log.info(
            'Set environment var %s to %r for %s process %d',
            CUDA_ENVVAR, os.environ[CUDA_ENVVAR], process_type, process_idx,
        )
        log.debug('Visible devices: %r', torch.cuda.device_count())

    return gpus_to_use


def cuda_envvars_for_policy(policy_id, process_type):
    set_gpus_for_process(policy_id, 1, process_type)


def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    stats = {f'memory_{process}': memory_mb}
    if device.type != 'cpu':
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update({f'gpu_mem_{process}': gpu_mem_mb, f'gpu_cache_{process}': gpu_cache_mb})

    return stats


def tensor_batch_size(tensor_batch):
    for _, _, v in iterate_recursively(tensor_batch):
        return v.shape[0]


class TensorBatcher:
    def __init__(self, batch_pool):
        self.batch_pool = batch_pool

    def cat(self, dict_of_tensor_arrays, macro_batch_size, use_pinned_memory, timing):
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
                cache_d[key] = torch.from_numpy(np.concatenate(tensor_arr, axis=0))
                if use_pinned_memory:
                    cache_d[key] = cache_d[key].pin_memory()
        else:
            with timing.add_time('batcher_mem'):
                for d1, cache_d, key, tensor_arr, cache_t in iter_dicts_recursively(dict_of_tensor_arrays, tensor_batch):
                    offset = 0
                    for t in tensor_arr:
                        first_dim = t.shape[0]
                        cache_t[offset:offset + first_dim].copy_(torch.as_tensor(t))
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
