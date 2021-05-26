import contextlib
import math
from typing import Optional, List

import faster_fifo
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
        self._tensors['rewards'].fill_(-42.42)  # if we're using uninitialized values it will be obvious
        self._tensors['dones'] = self.init_tensor(torch.bool, [1])
        self._tensors['dones'].fill_(True)
        self._tensors['policy_id'] = self.init_tensor(torch.int, [1])
        self._tensors['policy_id'].fill_(-1)  # -1 is an invalid policy index, experience from policy "-1" is always ignored

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

        queue_max_size_bytes = self.num_traj_buffers * 40  # 40 bytes to encode an int should be enough?
        self.free_buffers_queue = faster_fifo.Queue(max_size_bytes=queue_max_size_bytes)

        # since all buffers are initially free, we add all buffer indices to the queue
        self.free_buffers_queue.put_many_nowait([int(i) for i in np.arange(self.num_traj_buffers)])

    def calc_num_trajectory_buffers(self):
        """
        This calculates the number of shared trajectory (rollout) buffers required by the system to operate
        without interruptions.
        This consists of:
        1) at least one trajectory buffer per agent, such that we always have a location to save new experience
        2) a few trajectory buffers to hold data currently processed by the learner (including potential backlog)
        3) (potentially) some extra trajectory buffers to keep the system operational. These might be required
        i.e. in multi-agent envs when some agents are deactivated during the rollout. Such agents together with the
        learner can hold on to many buffers, such that active agents might not have enough free buffers to continue
        collecting the experience.
        """

        # Add a traj buffer for each agent
        num_traj_buffers = (self.cfg.num_workers + 1) * self.cfg.num_envs_per_worker * self.num_agents

        max_minibatches_to_accumulate = self.cfg.num_minibatches_to_accumulate
        if max_minibatches_to_accumulate == -1:
            # default value
            max_minibatches_to_accumulate = 2 * self.cfg.num_batches_per_iteration

        # Let each learner accumulate enough full sets of experience to pause learning
        max_experience_on_learners = max_minibatches_to_accumulate * self.cfg.batch_size * self.cfg.num_policies
        num_traj_buffers += max_experience_on_learners / self.cfg.rollout

        # Configurable excess ratio to be safe
        assert self.cfg.traj_buffers_excess_ratio >= 1.0
        num_traj_buffers = self.cfg.traj_buffers_excess_ratio * num_traj_buffers

        num_traj_buffers = int(math.ceil(num_traj_buffers))

        log.info('Using a total of %d trajectory buffers', num_traj_buffers)
        return num_traj_buffers

    def init_tensor(self, tensor_type, tensor_shape):
        if not isinstance(tensor_type, torch.dtype):
            tensor_type = to_torch_dtype(tensor_type)

        dimensions = [self.num_traj_buffers, self.cfg.rollout]
        final_shape = dimensions + list(tensor_shape)
        t = torch.zeros(final_shape, dtype=tensor_type)
        t.share_memory_()
        return t

    def tensor_dict_to_numpy(self):
        numpy_dict = copy_dict_structure(self._tensors)
        for d1, d2, key, curr_t, value2 in iter_dicts_recursively(self._tensors, numpy_dict):
            assert isinstance(curr_t, torch.Tensor)
            assert value2 is None
            d2[key] = curr_t.numpy()
            assert isinstance(d2[key], np.ndarray)
        return numpy_dict

    def get_trajectory_buffers(self, num_buffers: int, timing: Optional = None):
        """
        :param num_buffers: number of free buffer indices to obtain
        :param timing: for performance analysis
        :return: a list of indices of free buffers
        """
        indices: List[int] = []
        block = False

        while len(indices) < num_buffers:
            with timing.add_time('wait_buffers') if timing is not None else contextlib.suppress():
                try:
                    indices.extend(self.free_buffers_queue.get_many(
                        max_messages_to_get=num_buffers - len(indices), timeout=5, block=block,
                    ))
                except faster_fifo.Empty:
                    log.warning('Waiting for %d trajectory buffers...', num_buffers - len(indices))

                if len(indices) < num_buffers:
                    block = True

        return indices

    def free_trajectory_buffers(self, indices: List[int]):
        if len(indices) > 0:
            # Put ints as they are MUCH faster/smaller to pickle
            # Use the _int = int trick as calling int() in a tight loop is slow
            _int = int
            free_indices = [_int(i) for i in indices]
            self.free_buffers_queue.put_many_nowait(free_indices)


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
                n = new_data.cpu().numpy()
            elif isinstance(new_data, np.ndarray):
                n = new_data
            else:
                raise Exception(f'Type {type(new_data)} not supported in set_data_func')

            x[index] = n
