import os
import sys

from algorithms.appo.appo_utils import CUDA_ENVVAR
from algorithms.utils.arguments import maybe_load_from_checkpoint, get_algo_class, parse_args
from utils.get_available_gpus import get_available_gpus_without_triggering_pytorch_cuda_initialization
from utils.utils import log


def train(cfg):
    cfg = maybe_load_from_checkpoint(cfg)

    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.learn()
    algo.finalize()

    log.info('Done')
    return status


def main():
    """Script entry point."""
    available_gpus = get_available_gpus_without_triggering_pytorch_cuda_initialization(os.environ)
    if CUDA_ENVVAR not in os.environ:
        os.environ[CUDA_ENVVAR] = available_gpus
    os.environ[f'{CUDA_ENVVAR}_backup_'] = os.environ[CUDA_ENVVAR]
    os.environ[CUDA_ENVVAR] = ''

    cfg = parse_args()
    return train(cfg)


if __name__ == '__main__':
    sys.exit(main())
