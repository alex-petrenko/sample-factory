from __future__ import annotations

import math
from typing import List, Dict, Tuple

import torch
from gym import spaces
from torch import Tensor

from sample_factory.algo.sampling.rollout_worker import rollout_worker_device
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.multiprocessing_utils import get_queue
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import to_torch_dtype
from sample_factory.model.model_utils import get_hidden_size
from sample_factory.algo.utils.action_distributions import calc_num_actions, calc_num_logits
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import gpus_for_process
from sample_factory.utils.typing import PolicyID, MpQueue, Device
from sample_factory.utils.utils import log, AttrDict


def policy_device(cfg: AttrDict, policy_id: PolicyID) -> torch.device:
    """Inference/Learning device for the given policy."""

    if cfg.device == 'cpu':
        return torch.device('cpu')
    else:
        return torch.device('cuda', index=gpus_for_process(policy_id, 1)[0])


def init_tensor(leading_dimensions: List, tensor_type, tensor_shape, device: torch.device, share: bool) -> Tensor:
    if not isinstance(tensor_type, torch.dtype):
        tensor_type = to_torch_dtype(tensor_type)

    # filter out dimensions with size 0
    tensor_shape = [x for x in tensor_shape if x]

    final_shape = leading_dimensions + list(tensor_shape)
    t = torch.zeros(final_shape, dtype=tensor_type)

    if tensor_type in (torch.float, torch.float32, torch.float64):
        t.fill_(-42.42)
    elif tensor_type in (torch.int, torch.int32, torch.int64, torch.int8, torch.uint8):
        t.fill_(43)

    t = t.to(device)
    if share:
        t.share_memory_()

    return t


def action_info(env_info: EnvInfo) -> Tuple[int, int]:
    action_space = env_info.action_space
    num_actions = calc_num_actions(action_space)
    num_action_distribution_parameters = calc_num_logits(action_space)
    return num_actions, num_action_distribution_parameters


def policy_output_shapes(num_actions, num_action_distribution_parameters) -> List[Tuple[str, List]]:
    # policy outputs, this matches the expected output of the actor-critic
    policy_outputs = [
        ('actions', [num_actions]),
        ('action_logits', [num_action_distribution_parameters]),
        ('log_prob_actions', []),
        ('values', []),
        ('policy_version', []),
    ]
    return policy_outputs


def alloc_trajectory_tensors(env_info: EnvInfo, num_trajectories, rollout, hidden_size, device, share):
    obs_space = env_info.obs_space

    tensors = TensorDict()

    # just to minimize the amount of typing
    def init_trajectory_tensor(dtype_, shape_):
        return init_tensor([num_trajectories, rollout], dtype_, shape_, device, share)

    # policy inputs
    tensors['obs'] = TensorDict()
    if not isinstance(obs_space, spaces.Dict):
        raise Exception('Only Dict observations spaces are supported')

    # TODO: make sure we use this extra step at the end for value bootstrapping in the non batched runner
    for space_name, space in obs_space.spaces.items():
        tensors['obs'][space_name] = init_tensor([num_trajectories, rollout + 1], space.dtype, space.shape, device, share)
    tensors['rnn_states'] = init_tensor([num_trajectories, rollout + 1], torch.float32, [hidden_size], device, share)

    num_actions, num_action_distribution_parameters = action_info(env_info)
    policy_outputs = policy_output_shapes(num_actions, num_action_distribution_parameters)

    for name, shape in policy_outputs:
        assert name not in tensors
        tensors[name] = init_trajectory_tensor(torch.float32, shape)

    # env outputs
    tensors['rewards'] = init_trajectory_tensor(torch.float32, [])
    tensors['rewards'].fill_(-42.42)  # if we're using uninitialized values by mistake it will be obvious
    tensors['dones'] = init_trajectory_tensor(torch.bool, [])
    tensors['dones'].fill_(True)
    tensors['policy_id'] = init_trajectory_tensor(torch.int, [])
    tensors['policy_id'].fill_(-1)  # -1 is an invalid policy index, experience from policy "-1" is always ignored

    return tensors


def alloc_policy_output_tensors(cfg, env_info: EnvInfo, hidden_size, device, share):
    num_agents = env_info.num_agents
    envs_per_split = cfg.num_envs_per_worker // cfg.worker_num_splits

    policy_outputs_shape = [cfg.num_workers, cfg.worker_num_splits]
    if cfg.batched_sampling:
        policy_outputs_shape += [envs_per_split * num_agents]
    else:
        policy_outputs_shape += [envs_per_split, num_agents]

    num_actions, num_action_distribution_parameters = action_info(env_info)
    policy_outputs = policy_output_shapes(num_actions, num_action_distribution_parameters)
    policy_outputs += [('new_rnn_states', [hidden_size])]  # different name so we don't override current step rnn_state

    output_names, output_shapes = list(zip(*policy_outputs))
    output_sizes = [shape[0] if shape else 1 for shape in output_shapes]

    if cfg.batched_sampling:
        policy_output_tensors = TensorDict()
        for name, shape in policy_outputs:
            policy_output_tensors[name] = init_tensor(policy_outputs_shape, torch.float32, shape, device, share)
    else:
        # copying small policy outputs (e.g. individual value predictions & action logits) to shared memory is a
        # bottleneck on the policy worker. For optimization purposes we create additional tensors to hold
        # just concatenated policy outputs. Rollout workers parse the data and add it to the trajectory buffers
        # in a proper format
        outputs_combined_size = sum(output_sizes)
        policy_output_tensors = init_tensor(policy_outputs_shape, torch.float32, [outputs_combined_size], device, share)

    return policy_output_tensors, output_names, output_sizes


class BufferMgr(Configurable):
    def __init__(self, cfg, env_info: EnvInfo):
        super().__init__(cfg)
        self.env_info = env_info

        self.buffers_per_device: Dict[Device, int] = dict()

        for i in range(cfg.num_workers):
            # TODO: this should take into account whether we just need a GPU for sampling, or we actually receive observations on the GPU
            # otherwise it will not work for things like Megaverse or GPU-rendered DMLab
            sampling_device = str(rollout_worker_device(i, cfg))
            log.debug(f'Rollout worker {i} uses device {sampling_device}')

            num_buffers = self.env_info.num_agents * cfg.num_envs_per_worker
            buffers_for_device = self.buffers_per_device.get(sampling_device, 0) + num_buffers
            self.buffers_per_device[sampling_device] = buffers_for_device

        hidden_size = get_hidden_size(cfg)  # in case we have RNNs

        rollout = cfg.rollout
        self.trajectories_per_minibatch = cfg.batch_size // rollout
        self.trajectories_per_batch = cfg.num_batches_per_epoch * self.trajectories_per_minibatch

        if cfg.batched_sampling:
            worker_samples_per_iteration = (env_info.num_agents * cfg.num_envs_per_worker) // cfg.worker_num_splits
            assert math.gcd(self.trajectories_per_batch, worker_samples_per_iteration) == min(self.trajectories_per_batch, worker_samples_per_iteration), \
                f'{worker_samples_per_iteration=} should divide the {self.trajectories_per_batch=} or vice versa'
            self.worker_samples_per_iteration = worker_samples_per_iteration
        else:
            self.worker_samples_per_iteration = -1

        # TODO: need extra checks for sync RL, i.e. we should have enough buffers to feed the learner
        # i.e. 1 worker 10 envs with batch size of 32 trajectories does not work

        # TODO: another check for sync RL: what if we collect more experience per iteration than we can process in a training batch?

        share = not cfg.serial_mode

        if cfg.async_rl:
            # one set of buffers to sample, one to learn from. Coefficient 2 seems appropriate here.
            for device in self.buffers_per_device:
                self.buffers_per_device[device] *= 2
        else:
            # in synchronous mode we only allocate a single set of trajectories
            # and they are not released until the learner finishes learning from them
            pass

        # determine the number of minibatches we're allowed to accumulate before experience collection is halted
        self.max_batches_to_accumulate = cfg.num_batches_to_accumulate
        if not cfg.async_rl and self.max_batches_to_accumulate != 1:
            log.debug('In synchronous mode, we only accumulate one batch. Setting num_batches_to_accumulate to 1')
            self.max_batches_to_accumulate = 1

        # allocate trajectory buffers for sampling
        self.traj_buffer_queues: Dict[Device, MpQueue] = dict()
        self.traj_tensors_torch = dict()
        self.policy_output_tensors_torch = dict()

        for device, num_buffers in self.buffers_per_device.items():
            # make sure that at the very least we have enough buffers to feed the learner
            num_buffers = max(num_buffers, self.max_batches_to_accumulate * self.trajectories_per_batch * cfg.num_policies)

            self.traj_buffer_queues[device] = get_queue(cfg.serial_mode)

            self.traj_tensors_torch[device] = alloc_trajectory_tensors(
                self.env_info, num_buffers, rollout, hidden_size, device, share,
            )
            self.policy_output_tensors_torch[device], self.output_names, self.output_sizes = alloc_policy_output_tensors(
                cfg, self.env_info, hidden_size, device, share,
            )

            if cfg.batched_sampling:
                # big trajectory batches (slices) for batched sampling
                for i in range(0, num_buffers, self.worker_samples_per_iteration):
                    self.traj_buffer_queues[device].put(slice(i, i + self.worker_samples_per_iteration))
            else:
                # individual trajectories for more flexible non-batched sampling
                for i in range(num_buffers):
                    self.traj_buffer_queues[device].put(i)

        self.policy_versions = torch.zeros([cfg.num_policies], dtype=torch.int32)
        if share:
            self.policy_versions.share_memory_()
