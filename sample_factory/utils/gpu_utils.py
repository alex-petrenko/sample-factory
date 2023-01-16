import os
from typing import List, Optional

import torch

from sample_factory.utils.get_available_gpus import get_gpus_without_triggering_pytorch_cuda_initialization
from sample_factory.utils.utils import log

CUDA_ENVVAR = "CUDA_VISIBLE_DEVICES"


def set_global_cuda_envvars(cfg):
    if CUDA_ENVVAR not in os.environ:
        if cfg.device == "cpu":
            available_gpus = ""
        else:
            available_gpus = get_gpus_without_triggering_pytorch_cuda_initialization(os.environ)
        os.environ[CUDA_ENVVAR] = available_gpus
    log.info(f"Environment var {CUDA_ENVVAR} is {os.environ[CUDA_ENVVAR]}")


def get_available_gpus() -> List[int]:
    """
    Returns indices of GPUs specified by CUDA_VISIBLE_DEVICES.
    """
    orig_visible_devices = os.environ[f"{CUDA_ENVVAR}"]
    available_gpus = [int(g.strip()) for g in orig_visible_devices.split(",") if g and not g.isspace()]
    return available_gpus


def gpus_for_process(process_idx: int, num_gpus_per_process: int, gpu_mask: Optional[List[int]] = None) -> List[int]:
    """
    Returns indices of GPUs to use for a process. These indices already respect the CUDA_VISIBLE_DEVICES envvar.
    I.e. if CUDA_VISIBLE_DEVICES is '1,2,3', then from torch's there are three visible GPUs
    with indices 0, 1, and 2.
    Therefore, in this case gpus_for_process(0, 1) returns [0], gpus_for_process(1, 1) returns [1], etc.
    """

    available_gpus = get_available_gpus()
    if gpu_mask is not None:
        assert len(available_gpus) >= len(
            gpu_mask
        ), f"Number of available GPUs ({len(available_gpus)}) is less than number of GPUs in mask ({len(gpu_mask)})"
        available_gpus = [available_gpus[g] for g in gpu_mask]
    num_gpus = len(available_gpus)

    gpus_to_use = []
    if num_gpus == 0:
        return gpus_to_use

    first_gpu_idx = process_idx * num_gpus_per_process
    for i in range(num_gpus_per_process):
        index_mod_num_gpus = (first_gpu_idx + i) % num_gpus
        gpus_to_use.append(index_mod_num_gpus)

    log.debug(
        f"Using GPUs {gpus_to_use} for process {process_idx} (actually maps to GPUs {[available_gpus[g] for g in gpus_to_use]})"
    )
    return gpus_to_use


def set_gpus_for_process(process_idx, num_gpus_per_process, process_type, gpu_mask=None):
    # in this function we want to limit the number of GPUs visible to the process, i.e. if
    # CUDA_VISIBLE_DEVICES is '1,2,3' and we want to use GPU index 2, then we want to set
    # CUDA_VISIBLE_DEVICES to '3' for this process
    gpus_to_use = gpus_for_process(process_idx, num_gpus_per_process, gpu_mask)

    if not gpus_to_use:
        os.environ[CUDA_ENVVAR] = ""
        log.debug("Not using GPUs for %s process %d", process_type, process_idx)
    else:
        available_gpus = get_available_gpus()
        cuda_devices_to_use = ",".join([str(available_gpus[g]) for g in gpus_to_use])
        os.environ[CUDA_ENVVAR] = cuda_devices_to_use
        log.info(
            "Set environment var %s to %r (GPU indices %r) for %s process %d",
            CUDA_ENVVAR,
            os.environ[CUDA_ENVVAR],
            gpus_to_use,
            process_type,
            process_idx,
        )
        log.debug("Num visible devices: %r", torch.cuda.device_count())

    return gpus_to_use


# TODO: do we need this func?
def cuda_envvars_for_policy(policy_id, process_type):
    set_gpus_for_process(policy_id, 1, process_type)
