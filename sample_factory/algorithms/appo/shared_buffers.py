import math

import numpy as np
import torch
from gym import spaces

from sample_factory.algorithms.appo.appo_utils import copy_dict_structure, iter_dicts_recursively, iterate_recursively
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import calc_num_logits, calc_num_actions
from sample_factory.utils.utils import log


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def to_numpy(t, num_dimensions):
    arr_shape = t.shape[:num_dimensions]
    arr = np.ndarray(arr_shape, dtype=object)
    to_numpy_func(t, arr)
    return arr


def to_numpy_func(t, arr):
    if len(arr.shape) == 1:
        for i in range(t.shape[0]):
            arr[i] = t[i]
    else:
        for i in range(t.shape[0]):
            to_numpy_func(t[i], arr[i])


def ensure_memory_shared(*tensors):
    """To prevent programming errors, ensure all tensors are in shared memory."""
    for tensor_dict in tensors:
        for _, _, t in iterate_recursively(tensor_dict):
            assert t.is_shared()


class PolicyOutput:
    def __init__(self, name, size):
        self.name = name
        self.size = size


class SharedBuffers:
    def __init__(self, cfg, num_agents, obs_space, action_space):
        self.cfg = cfg
        self.num_agents = num_agents
        self.envs_per_split = cfg.num_envs_per_worker // cfg.worker_num_splits
        self.num_traj_buffers = self.calc_num_trajectory_buffers()

        num_actions = calc_num_actions(action_space)
        num_action_logits = calc_num_logits(action_space)

        hidden_size = get_hidden_size(self.cfg)

        log.debug('Allocating shared memory for trajectories')
        self._tensors = TensorDict()

        # policy inputs
        obs_dict = TensorDict()
        self._tensors['obs'] = obs_dict
        if isinstance(obs_space, spaces.Dict):
            for space_name, space in obs_space.spaces.items():
                obs_dict[space_name] = self.init_tensor(space.dtype, space.shape)
        else:
            raise Exception('Only Dict observations spaces are supported')

        # env outputs
        self._tensors['rewards'] = self.init_tensor(torch.float32, [1])
        self._tensors['dones'] = self.init_tensor(torch.bool, [1])

        # policy outputs
        policy_outputs = [
            ('actions', num_actions),
            ('action_logits', num_action_logits),
            ('log_prob_actions', 1),
            ('values', 1),
            ('policy_version', 1),
            ('rnn_states', hidden_size)
        ]

        policy_outputs = [PolicyOutput(*po) for po in policy_outputs]
        policy_outputs = sorted(policy_outputs, key=lambda policy_output: policy_output.name)

        for po in policy_outputs:
            self._tensors[po.name] = self.init_tensor(torch.float32, [po.size])

        ensure_memory_shared(self._tensors)

        # this is for performance optimization
        # indexing in numpy arrays is faster than in PyTorch tensors
        self.tensors = self.tensor_dict_to_numpy()

        # create a shared tensor to indicate when the learner is done with the trajectory buffer and
        # it can be used to store the next trajectory
        traj_buffer_available_shape = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            self.num_traj_buffers,
        ]
        self._is_traj_tensor_available = torch.ones(traj_buffer_available_shape, dtype=torch.uint8)
        self._is_traj_tensor_available.share_memory_()
        self.is_traj_tensor_available = self._is_traj_tensor_available.numpy()

        # copying small policy outputs (e.g. individual value predictions & action logits) to shared memory is a
        # bottleneck on the policy worker. For optimization purposes we create additional tensors to hold
        # just concatenated policy outputs. Rollout workers parse the data and add it to the trajectory buffers
        # in a proper format
        policy_outputs_combined_size = sum(po.size for po in policy_outputs)
        policy_outputs_shape = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            policy_outputs_combined_size,
        ]

        self.policy_outputs = policy_outputs
        self._policy_output_tensors = torch.zeros(policy_outputs_shape, dtype=torch.float32)
        self._policy_output_tensors.share_memory_()
        self.policy_output_tensors = self._policy_output_tensors.numpy()

        self._policy_versions = torch.zeros([self.cfg.num_policies], dtype=torch.int32)
        self._policy_versions.share_memory_()
        self.policy_versions = self._policy_versions.numpy()

        # a list of boolean flags to be shared among components that indicate that experience collection should be
        # temporarily stopped (e.g. due to too much experience accumulated on the learner)
        self._stop_experience_collection = torch.ones([self.cfg.num_policies], dtype=torch.bool)
        self._stop_experience_collection.share_memory_()
        self.stop_experience_collection = self._stop_experience_collection.numpy()

    def calc_num_trajectory_buffers(self):
        # calculate how many buffers are required per env runner to collect one "macro batch" for training
        # once macro batch is collected, all buffers will be released
        # we could have just copied the tensors on the learner to avoid this complicated logic, but it's better for
        # performance to keep data in shared buffers until they're needed
        samples_per_iteration = self.cfg.num_batches_per_iteration * self.cfg.batch_size * self.cfg.num_policies
        num_traj_buffers = samples_per_iteration / (self.cfg.num_workers * self.cfg.num_envs_per_worker * self.num_agents * self.cfg.rollout)

        # make sure we definitely have enough buffers to actually never wait
        # usually it'll be just two buffers and we swap back and forth
        num_traj_buffers *= 3

        # make sure we have at least two to swap between so we never actually have to wait
        num_traj_buffers = math.ceil(max(num_traj_buffers, self.cfg.min_traj_buffers_per_worker))
        log.info('Using %d sets of trajectory buffers', num_traj_buffers)
        return num_traj_buffers

    def init_tensor(self, tensor_type, tensor_shape):
        if not isinstance(tensor_type, torch.dtype):
            tensor_type = to_torch_dtype(tensor_type)

        dimensions = self.tensor_dimensions()
        final_shape = dimensions + list(tensor_shape)
        t = torch.zeros(final_shape, dtype=tensor_type)
        t.share_memory_()
        return t

    def tensor_dimensions(self):
        dimensions = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            self.num_traj_buffers,
            self.cfg.rollout,
        ]
        return dimensions

    def tensor_dict_to_numpy(self):
        numpy_dict = copy_dict_structure(self._tensors)
        for d1, d2, key, curr_t, value2 in iter_dicts_recursively(self._tensors, numpy_dict):
            assert isinstance(curr_t, torch.Tensor)
            assert value2 is None
            d2[key] = curr_t.numpy()
            assert isinstance(d2[key], np.ndarray)
        return numpy_dict


class TensorDict(dict):
    def index(self, indices):
        return self.index_func(self, indices)

    def index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self.index_func(value, indices)
            return res
        else:
            t = x[indices]
            return t

    def set_data(self, index, new_data):
        self.set_data_func(self, index, new_data)

    def set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self.set_data_func(x[new_data_key], index, new_data_value)
        if torch.is_tensor(x):
            if isinstance(new_data, torch.Tensor):
                t = new_data
            elif isinstance(new_data, np.ndarray):
                t = torch.from_numpy(new_data)
            else:
                raise Exception(f'Type {type(new_data)} not supported in set_data_func')

            x[index].copy_(t)

        elif isinstance(x, np.ndarray):
            if isinstance(new_data, torch.Tensor):
                n = new_data.numpy()
            elif isinstance(new_data, np.ndarray):
                n = new_data
            else:
                raise Exception(f'Type {type(new_data)} not supported in set_data_func')

            x[index] = n
