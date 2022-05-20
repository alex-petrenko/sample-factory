from typing import Tuple, List, Dict, Any, Optional

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import AttrDict


class VectorEnvRunner(Configurable):
    def __init__(self, cfg: AttrDict, env_info: EnvInfo, worker_idx, split_idx, buffer_mgr, sampling_device: str):
        super().__init__(cfg)
        self.env_info: EnvInfo = env_info

        self.worker_idx = worker_idx
        self.split_idx = split_idx

        self.rollout_step: int = 0  # current position in the rollout across all envs
        self.env_step_ready = False

        self.buffer_mgr = buffer_mgr
        self.traj_tensors = buffer_mgr.traj_tensors[sampling_device]
        self.policy_output_tensors = buffer_mgr.policy_output_tensors[sampling_device][worker_idx, split_idx]
        self.traj_buffer_queue = buffer_mgr.traj_buffer_queues[sampling_device]

    def init(self, timing) -> Dict:
        raise NotImplementedError()

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError()

    def update_trajectory_buffers(self, timing) -> bool:
        raise NotImplementedError()

    def generate_policy_request(self, timing) -> Optional[Dict]:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()
