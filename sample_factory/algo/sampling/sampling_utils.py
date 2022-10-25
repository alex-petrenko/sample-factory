from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.gpu_utils import gpus_for_process
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID


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

    def init(self, timing: Timing):
        raise NotImplementedError()

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError()

    def update_trajectory_buffers(self, timing) -> bool:
        raise NotImplementedError()

    def generate_policy_request(self) -> Optional[Dict]:
        raise NotImplementedError()

    def synchronize_devices(self) -> None:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


def rollout_worker_device(worker_idx, cfg: AttrDict, env_info: EnvInfo) -> torch.device:
    # TODO: this should correspond to whichever device we have observations on, not just whether we use this device at all
    # TODO: test with Megaverse on a multi-GPU system
    # TODO: actions on a GPU device? Convert to CPU for some envs?

    if not env_info.gpu_observations:
        return torch.device("cpu")
    gpus_to_use = gpus_for_process(worker_idx, num_gpus_per_process=1, gpu_mask=cfg.actor_worker_gpus)
    assert len(gpus_to_use) <= 1
    sampling_device = torch.device("cuda", index=gpus_to_use[0]) if gpus_to_use else torch.device("cpu")
    return sampling_device


def record_episode_statistics_wrapper_stats(info: Dict) -> Optional[Tuple[float, float]]:
    """
    Some envs like Atari use a special wrapper gym.wrappers.RecordEpisodeStatistics to record episode stats.
    This accounts for things like reward clipping in the wrappers or frameskip affecting length.
    """
    if ep_info := info.get("episode", None):
        return ep_info["r"], ep_info["l"]
    return None
