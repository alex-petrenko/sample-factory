import torch

from sample_factory.utils.utils import memory_consumption_mb


EPS = 1e-8


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


# custom experiments can define functions to this list to do something extra with the raw episode summaries
# coming from the environments
EXTRA_EPISODIC_STATS_PROCESSING = []

# custom experiments or environments can append functions to this list to postprocess some summaries, or aggregate
# summaries, or do whatever else the user wants
EXTRA_PER_POLICY_SUMMARIES = []

def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    stats = {f'memory_{process}': memory_mb}
    if device.type != 'cpu':
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update({f'gpu_mem_{process}': gpu_mem_mb, f'gpu_cache_{process}': gpu_cache_mb})

    return stats


# TODO: do we need this?
def num_env_steps(infos):
    """Calculate number of environment frames in a batch of experience."""

    total_num_frames = 0
    for info in infos:
        total_num_frames += info.get('num_frames', 1)
    return total_num_frames
