import torch.multiprocessing as mp
from multiprocessing.context import BaseContext
from typing import Optional
import torch
import os
from sample_factory.utils.utils import static_vars


@static_vars(mp_ctx=None)
def get_mp_ctx(serial: bool) -> Optional[BaseContext]:
    if serial:
        return None

    if get_mp_ctx.mp_ctx is None:
        get_mp_ctx.mp_ctx = mp.get_context("spawn")
    return get_mp_ctx.mp_ctx


def get_lock(serial=False, mp_ctx=None):
    if serial:
        return FakeLock()
    else:
        return get_mp_lock(mp_ctx)


def get_mp_lock(mp_ctx: Optional[BaseContext] = None):
    lock_cls = mp.Lock if mp_ctx is None else mp_ctx.Lock
    return lock_cls()


class FakeLock:
    def acquire(self, *args, **kwargs):
        pass

    def release(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

def init_process_dist(rank: int, size: int, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)