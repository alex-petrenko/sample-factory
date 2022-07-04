import torch

from sample_factory.utils.utils import memory_consumption_mb


EPS = 1e-8


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


def memory_stats(process, device):
    memory_mb = memory_consumption_mb()
    stats = {f'memory_{process}': memory_mb}
    if device.type != 'cpu':
        gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
        gpu_cache_mb = torch.cuda.memory_reserved(device) / 1e6
        stats.update({f'gpu_mem_{process}': gpu_mem_mb, f'gpu_cache_{process}': gpu_cache_mb})

    return stats
