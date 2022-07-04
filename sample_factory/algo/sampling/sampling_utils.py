from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from torch import Tensor

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.configurable import Configurable
from sample_factory.envs.env_wrappers import TimeLimitWrapper
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import AttrDict

# "TimeLimit.truncated" is the key used by Gym TimeLimit wrapper.
# "time_outs" is used by IsaacGym.
TIMEOUT_KEYS: Tuple = ("time_outs", TimeLimitWrapper.terminated_by_timer)


class VectorEnvRunner(Configurable):
    def __init__(self, cfg: AttrDict, env_info: EnvInfo, worker_idx, split_idx, buffer_mgr, sampling_device: str):
        super().__init__(cfg)
        self.env_info: EnvInfo = env_info

        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step: int = 0  # current position in the rollout across all envs
        self.env_step_ready = False

        self.buffer_mgr = buffer_mgr
        self.traj_buffer_queue = buffer_mgr.traj_buffer_queues[sampling_device]
        self.traj_tensors = buffer_mgr.traj_tensors_torch[sampling_device]
        self.policy_output_tensors = buffer_mgr.policy_output_tensors_torch[sampling_device][worker_idx, split_idx]

    def init(self, timing) -> Dict:
        raise NotImplementedError()

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError()

    def update_trajectory_buffers(self, timing, block=False) -> bool:
        raise NotImplementedError()

    def generate_policy_request(self, timing) -> Optional[Dict]:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


def fix_action_shape(actions: Tensor | np.ndarray, integer_actions: bool) -> Tensor | np.ndarray:
    if actions.ndim == 0:
        if integer_actions:
            actions = actions.item()
        else:
            # envs with continuous actions typically expect a vector of actions (i.e. Mujoco)
            # if there's only one action (i.e. Mujoco pendulum) then we need to make it a 1D vector
            actions = unsqueeze_tensor(actions, -1)

    return actions
