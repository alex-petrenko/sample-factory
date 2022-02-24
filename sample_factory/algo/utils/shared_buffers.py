import math

import torch
from gym import spaces

from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.appo.shared_buffers import TensorDict, to_torch_dtype
from sample_factory.algorithms.utils.action_distributions import calc_num_actions, calc_num_logits
from sample_factory.cfg.configurable import Configurable


def init_trajectory_tensor(num_trajectories, rollout_len, tensor_type, tensor_shape, device):
    if not isinstance(tensor_type, torch.dtype):
        tensor_type = to_torch_dtype(tensor_type)

    dimensions = [num_trajectories, rollout_len]

    # filter out dimensions with size 0
    tensor_shape = [x for x in tensor_shape if x]

    final_shape = dimensions + list(tensor_shape)
    t = torch.zeros(final_shape, dtype=tensor_type)

    if tensor_type in (torch.float, torch.float32, torch.float64):
        t.fill_(-42.42)
    elif tensor_type in (torch.int, torch.int32, torch.int64, torch.int8, torch.uint8):
        t.fill_(43)

    t = t.to(device)
    return t


# TODO: remove the code duplication!
def allocate_trajectory_buffers(env_info, num_trajectories, rollout, hidden_size, device):
    obs_space = env_info.obs_space
    action_space = env_info.action_space
    num_actions = calc_num_actions(action_space)
    num_action_logits = calc_num_logits(action_space)

    tensors = TensorDict()

    # just to minimize the amount of typing
    def init_tensor(dtype_, shape_):
        return init_trajectory_tensor(num_trajectories, rollout, dtype_, shape_, device)

    # policy inputs
    obs_dict = TensorDict()
    tensors['obs'] = obs_dict
    if isinstance(obs_space, spaces.Dict):
        for space_name, space in obs_space.spaces.items():
            obs_dict[space_name] = init_tensor(space.dtype, space.shape)
    else:
        raise Exception('Only Dict observations spaces are supported')

    # env outputs
    tensors['rewards'] = init_tensor(torch.float32, [])
    tensors['rewards'].fill_(-42.42)  # if we're using uninitialized values by mistake it will be obvious
    tensors['dones'] = init_tensor(torch.bool, [])
    tensors['dones'].fill_(True)
    tensors['policy_id'] = init_tensor(torch.int, [])
    tensors['policy_id'].fill_(-1)  # -1 is an invalid policy index, experience from policy "-1" is always ignored

    # policy outputs, this matches the expected output of the actor-critic
    policy_outputs = [
        ('actions', [num_actions]),
        ('action_logits', [num_action_logits]),
        ('log_prob_actions', []),
        ('values', []),
        ('policy_version', []),
        ('rnn_states', [hidden_size])
    ]

    for name, shape in policy_outputs:
        assert name not in tensors
        tensors[name] = init_tensor(torch.float32, shape)

    return tensors


class BufferMgr(Configurable):
    def __init__(self, cfg, env_info, device):
        super().__init__(cfg)
        self.env_info = env_info

        hidden_size = get_hidden_size(self.cfg)  # in case we have RNNs

        rollout = self.cfg.rollout
        self.trajectories_per_minibatch = self.cfg.batch_size // rollout
        self.trajectories_per_batch = self.cfg.num_batches_per_iteration * self.trajectories_per_minibatch

        assert math.gcd(self.trajectories_per_batch, self.env_info.num_agents), \
            'Num agents should divide the number of trajectories per batch or vice versa (for performance reasons)'

        self.total_num_trajectories = max(self.env_info.num_agents, self.trajectories_per_batch)
        self.traj_tensors = allocate_trajectory_buffers(
            self.env_info, self.total_num_trajectories, rollout, hidden_size, device,
        )

        # TODO: share memory for async algorithms?
        self.policy_versions = torch.zeros([self.cfg.num_policies], dtype=torch.int32)
