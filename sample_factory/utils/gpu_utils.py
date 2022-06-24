import os

import torch

from sample_factory.utils.get_available_gpus import get_gpus_without_triggering_pytorch_cuda_initialization
from sample_factory.utils.utils import log

CUDA_ENVVAR = 'CUDA_VISIBLE_DEVICES'


def set_global_cuda_envvars(cfg):
    if CUDA_ENVVAR not in os.environ:
        if cfg.device == 'cpu':
            available_gpus = ''
        else:
            available_gpus = get_gpus_without_triggering_pytorch_cuda_initialization(os.environ)
        os.environ[CUDA_ENVVAR] = available_gpus


def get_available_gpus():
    orig_visible_devices = os.environ[f'{CUDA_ENVVAR}']
    available_gpus = [int(g) for g in orig_visible_devices.split(',') if g]
    return available_gpus


def gpus_for_process(process_idx, num_gpus_per_process, gpu_mask=None):
    available_gpus = get_available_gpus()
    if gpu_mask is not None:
        assert len(available_gpus) >= len(available_gpus)
        available_gpus = [available_gpus[g] for g in gpu_mask]
    num_gpus = len(available_gpus)

    gpus_to_use = []
    if num_gpus == 0:
        return gpus_to_use

    first_gpu_idx = process_idx * num_gpus_per_process
    for i in range(num_gpus_per_process):
        index_mod_num_gpus = (first_gpu_idx + i) % num_gpus
        gpus_to_use.append(available_gpus[index_mod_num_gpus])
    return gpus_to_use


def set_gpus_for_process(process_idx, num_gpus_per_process, process_type, gpu_mask=None):
    gpus_to_use = gpus_for_process(process_idx, num_gpus_per_process, gpu_mask)

    if not gpus_to_use:
        os.environ[CUDA_ENVVAR] = ''
        log.debug('Not using GPUs for %s process %d', process_type, process_idx)
    else:
        os.environ[CUDA_ENVVAR] = ','.join([str(g) for g in gpus_to_use])
        log.info(
            'Set environment var %s to %r for %s process %d',
            CUDA_ENVVAR, os.environ[CUDA_ENVVAR], process_type, process_idx,
        )
        log.debug('Visible devices: %r', torch.cuda.device_count())

    return gpus_to_use


# TODO: do we need this func?
def cuda_envvars_for_policy(policy_id, process_type):
    set_gpus_for_process(policy_id, 1, process_type)