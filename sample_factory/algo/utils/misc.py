import torch

from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import memory_consumption_mb

EPS = 1e-8


# stats dictionary keys
EPISODIC = "episodic"
LEARNER_ENV_STEPS = "learner_env_steps"
TRAIN_STATS = "train"
TIMING_STATS = "timing"
STATS_KEY = "stats"
SAMPLES_COLLECTED = "samples_collected"
POLICY_ID_KEY = "policy_id"


MAGIC_FLOAT = -4242.42
MAGIC_INT = 43


# custom signal names
def new_trajectories_signal(policy_id: PolicyID) -> str:
    return f"p{policy_id}_trajectories"


def advance_rollouts_signal(rollout_worker_idx: int) -> str:
    return f"advance{rollout_worker_idx}"


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    stats = {f"memory_{process}": memory_mb}
    if device.type != "cpu":
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update({f"gpu_mem_{process}": gpu_mem_mb, f"gpu_cache_{process}": gpu_cache_mb})

    return stats
