import math
from typing import List, Dict

import torch
from gym import spaces
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.queues import get_mp_queue
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.appo.shared_buffers import TensorDict, to_torch_dtype
from sample_factory.algorithms.utils.action_distributions import calc_num_actions, calc_num_logits
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import PolicyID, MpQueue
from sample_factory.utils.utils import log


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


# TODO: remove the code duplication!
def allocate_tensors(cfg, env_info: EnvInfo, num_trajectories, rollout, hidden_size, device, share):
    num_agents = env_info.num_agents
    envs_per_split = cfg.num_envs_per_worker // cfg.worker_num_splits

    obs_space = env_info.obs_space
    action_space = env_info.action_space
    num_actions = calc_num_actions(action_space)
    num_action_logits = calc_num_logits(action_space)

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

    # policy outputs, this matches the expected output of the actor-critic
    policy_outputs = [
        ('actions', [num_actions]),
        ('action_logits', [num_action_logits]),
        ('log_prob_actions', []),
        ('values', []),
        ('policy_version', []),
    ]

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

    # TODO: implement this
    # copying small policy outputs (e.g. individual value predictions & action logits) to shared memory is a
    # bottleneck on the policy worker. For optimization purposes we create additional tensors to hold
    # just concatenated policy outputs. Rollout workers parse the data and add it to the trajectory buffers
    # in a proper format
    # policy_outputs_combined_size = sum(po.size for po in policy_outputs)
    # policy_outputs_shape = [
    #     self.cfg.num_workers,
    #     self.cfg.worker_num_splits,
    #     self.envs_per_split,
    #     self.num_agents,
    #     policy_outputs_combined_size,
    # ]
    # self.policy_outputs = policy_outputs
    # self._policy_output_tensors = torch.zeros(policy_outputs_shape, dtype=torch.float32)
    # self._policy_output_tensors.share_memory_()
    # self.policy_output_tensors = self._policy_output_tensors.numpy()

    # TODO: this only currently works with contiguous sampling
    policy_outputs_shape = [
        cfg.num_workers,
        cfg.worker_num_splits,
        envs_per_split,
        num_agents,
    ]

    # TODO: version for non-contiguous sampler
    policy_output_tensors = TensorDict()
    policy_outputs += [('new_rnn_states', [hidden_size])]  # different name so we don't override current step rnn_state
    for name, shape in policy_outputs:
        policy_output_tensors[name] = init_tensor(policy_outputs_shape, torch.float32, shape, device, share)

    return tensors, policy_output_tensors


class BufferMgr(Configurable):
    def __init__(self, cfg, env_info):
        super().__init__(cfg)
        self.env_info = env_info

        self.traj_buffer_queues: Dict[PolicyID, MpQueue] = {p: get_mp_queue() for p in range(cfg.num_policies)}

        # TODO: do not initialize CUDA in the main process if we can?
        policy_id = 0  # TODO: multi-policy case
        device_idx = policy_id % torch.cuda.device_count()
        self.device = torch.device('cuda', index=device_idx)

        hidden_size = get_hidden_size(self.cfg)  # in case we have RNNs

        rollout = self.cfg.rollout
        self.trajectories_per_minibatch = self.cfg.batch_size // rollout
        self.trajectories_per_batch = self.cfg.num_batches_per_iteration * self.trajectories_per_minibatch

        assert math.gcd(self.trajectories_per_batch, self.env_info.num_agents) == min(self.trajectories_per_batch, self.env_info.num_agents), \
            'Num agents should divide the number of trajectories per batch or vice versa (for performance reasons)'

        share = not cfg.serial_mode

        sampling_trajectories = self.env_info.num_agents
        if self.cfg.async_rl:
            sampling_trajectories *= 2  # TODO

        self.total_num_trajectories = max(sampling_trajectories, self.trajectories_per_batch)

        self.traj_tensors, self.policy_output_tensors = allocate_tensors(
            self.cfg, self.env_info, self.total_num_trajectories, rollout, hidden_size, self.device, share,
        )

        # TODO: numpy tensor stuff
        # TODO: for non-contiguous sampler we should do the batched policy outputs trick
        self.policy_versions = torch.zeros([self.cfg.num_policies], dtype=torch.int32)
        if share:
            self.policy_versions.share_memory_()
