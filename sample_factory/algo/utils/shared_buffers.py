from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from gym import spaces
from signal_slot.queue_utils import get_queue
from torch import Tensor

from sample_factory.algo.sampling.sampling_utils import rollout_worker_device
from sample_factory.algo.utils.action_distributions import calc_num_action_parameters, calc_num_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import MAGIC_FLOAT, MAGIC_INT
from sample_factory.algo.utils.rl_utils import trajectories_per_training_iteration
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import to_torch_dtype
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.gpu_utils import gpus_for_process
from sample_factory.utils.typing import Device, MpQueue, PolicyID
from sample_factory.utils.utils import log


def policy_device(cfg: AttrDict, policy_id: PolicyID) -> torch.device:
    """Inference/Learning device for the given policy."""

    if cfg.device == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda", index=gpus_for_process(policy_id, 1)[0])


def init_tensor(leading_dimensions: List, tensor_type, tensor_shape, device: torch.device, share: bool) -> Tensor:
    if not isinstance(tensor_type, torch.dtype):
        tensor_type = to_torch_dtype(tensor_type)

    # filter out dimensions with size 0
    tensor_shape = [x for x in tensor_shape if x]

    final_shape = leading_dimensions + list(tensor_shape)
    t = torch.zeros(final_shape, dtype=tensor_type)

    # fill with magic values to make it easy to spot if we ever use unintialized data
    if t.is_floating_point():
        t.fill_(MAGIC_FLOAT)
    elif tensor_type in (torch.int, torch.int32, torch.int64, torch.int8, torch.uint8):
        t.fill_(MAGIC_INT)

    t = t.to(device)

    # CUDA tensors are already shared by default
    if share and not t.is_cuda:
        t.share_memory_()

    return t


def action_info(env_info: EnvInfo) -> Tuple[int, int]:
    action_space = env_info.action_space
    num_actions = calc_num_actions(action_space)
    num_action_distribution_parameters = calc_num_action_parameters(action_space)
    return num_actions, num_action_distribution_parameters


def policy_output_shapes(num_actions, num_action_distribution_parameters) -> List[Tuple[str, List]]:
    # policy outputs, this matches the expected output of the actor-critic
    policy_outputs = [
        ("actions", [num_actions]),
        ("action_logits", [num_action_distribution_parameters]),
        ("log_prob_actions", []),
        ("values", []),
        ("policy_version", []),
    ]
    return policy_outputs


def alloc_trajectory_tensors(env_info: EnvInfo, num_traj, rollout, rnn_size, device, share) -> TensorDict:
    obs_space = env_info.obs_space

    tensors = TensorDict()

    # policy inputs
    tensors["obs"] = TensorDict()
    if not isinstance(obs_space, spaces.Dict):
        raise Exception("Only Dict observations spaces are supported")

    # we need to allocate an extra rollout step here to calculate the value estimates for the last step
    for space_name, space in obs_space.spaces.items():
        tensors["obs"][space_name] = init_tensor([num_traj, rollout + 1], space.dtype, space.shape, device, share)
    tensors["rnn_states"] = init_tensor([num_traj, rollout + 1], torch.float32, [rnn_size], device, share)

    num_actions, num_action_distribution_parameters = action_info(env_info)
    policy_outputs = policy_output_shapes(num_actions, num_action_distribution_parameters)

    # we need one more step to hold values for the last step
    outputs_with_extra_rollout_step = ["values"]

    for name, shape in policy_outputs:
        assert name not in tensors
        rollout_len = rollout + 1 if name in outputs_with_extra_rollout_step else rollout
        tensors[name] = init_tensor([num_traj, rollout_len], torch.float32, shape, device, share)

    # env outputs
    tensors["rewards"] = init_tensor([num_traj, rollout], torch.float32, [], device, share)
    tensors["rewards"].fill_(-42.42)  # if we're using uninitialized values by mistake it will be obvious
    tensors["dones"] = init_tensor([num_traj, rollout], torch.bool, [], device, share)
    tensors["dones"].fill_(True)
    tensors["time_outs"] = init_tensor([num_traj, rollout], torch.bool, [], device, share)
    tensors["time_outs"].fill_(False)  # no timeouts by default
    tensors["policy_id"] = init_tensor([num_traj, rollout], torch.int, [], device, share)
    tensors["policy_id"].fill_(-1)  # -1 is an invalid policy index, experience from policy "-1" is always ignored
    tensors["valids"] = init_tensor([num_traj, rollout + 1], torch.bool, [], device, share)
    tensors["valids"].fill_(False)  # no valid experience by default

    return tensors


def alloc_policy_output_tensors(cfg, env_info: EnvInfo, rnn_size, device, share):
    num_agents = env_info.num_agents
    envs_per_split = cfg.num_envs_per_worker // cfg.worker_num_splits

    policy_outputs_shape = [cfg.num_workers, cfg.worker_num_splits]
    if cfg.batched_sampling:
        policy_outputs_shape += [envs_per_split * num_agents]
    else:
        policy_outputs_shape += [envs_per_split, num_agents]

    num_actions, num_action_distribution_parameters = action_info(env_info)
    policy_outputs = policy_output_shapes(num_actions, num_action_distribution_parameters)
    policy_outputs += [("new_rnn_states", [rnn_size])]  # different name so we don't override current step rnn_state

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
            sampling_device = str(rollout_worker_device(i, cfg, self.env_info))
            log.debug(f"Rollout worker {i} uses device {sampling_device}")

            num_buffers = env_info.num_agents * cfg.num_envs_per_worker
            buffers_for_device = self.buffers_per_device.get(sampling_device, 0) + num_buffers
            self.buffers_per_device[sampling_device] = buffers_for_device

        rnn_size = get_rnn_size(cfg)  # in case we have RNNs

        self.trajectories_per_training_iteration = trajectories_per_training_iteration(cfg)

        if cfg.batched_sampling:
            worker_traj_per_iteration = (env_info.num_agents * cfg.num_envs_per_worker) // cfg.worker_num_splits
            assert math.gcd(self.trajectories_per_training_iteration, worker_traj_per_iteration) == min(
                self.trajectories_per_training_iteration, worker_traj_per_iteration
            ), f"{worker_traj_per_iteration=} should divide the {self.trajectories_per_training_iteration=} or vice versa"
            self.sampling_trajectories_per_iteration = worker_traj_per_iteration
        else:
            self.sampling_trajectories_per_iteration = -1

        share = not cfg.serial_mode

        if cfg.async_rl or cfg.num_policies > 1:
            # One set of buffers to sample, one to learn from. Coefficient 2 seems appropriate here.
            # Also: multi-policy training may require more buffers since some trajectories need to be sent
            # to multiple workers.
            for device in self.buffers_per_device:
                self.buffers_per_device[device] *= 2
        else:
            # in synchronous mode we only allocate a single set of trajectories
            # and they are not released until the learner finishes learning from them
            pass

        # determine the number of minibatches we're allowed to accumulate before experience collection is halted
        self.max_batches_to_accumulate = cfg.num_batches_to_accumulate
        if not cfg.async_rl:
            log.debug("In synchronous mode, we only accumulate one batch. Setting num_batches_to_accumulate to 1")
            self.max_batches_to_accumulate = 1

        # allocate trajectory buffers for sampling
        self.traj_buffer_queues: Dict[Device, MpQueue] = dict()
        self.traj_tensors_torch = dict()
        self.policy_output_tensors_torch = dict()

        for device, num_buffers in self.buffers_per_device.items():
            # make sure that at the very least we have enough buffers to feed the learner
            num_buffers = max(
                num_buffers,
                self.max_batches_to_accumulate * self.trajectories_per_training_iteration * cfg.num_policies,
            )

            self.traj_buffer_queues[device] = get_queue(cfg.serial_mode)

            self.traj_tensors_torch[device] = alloc_trajectory_tensors(
                env_info,
                num_buffers,
                cfg.rollout,
                rnn_size,
                device,
                share,
            )
            self.policy_output_tensors_torch[device], output_names, output_sizes = alloc_policy_output_tensors(
                cfg, env_info, rnn_size, device, share
            )
            self.output_names, self.output_sizes = output_names, output_sizes

            if cfg.batched_sampling:
                # big trajectory batches (slices) for batched sampling
                for i in range(0, num_buffers, self.sampling_trajectories_per_iteration):
                    self.traj_buffer_queues[device].put(slice(i, i + self.sampling_trajectories_per_iteration))
            else:
                # individual trajectories for more flexible non-batched sampling
                for i in range(num_buffers):
                    self.traj_buffer_queues[device].put(i)

        self.policy_versions = torch.zeros([cfg.num_policies], dtype=torch.int32)
        if share:
            self.policy_versions.share_memory_()
