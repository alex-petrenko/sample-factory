import sys

from algorithms.appo.appo_utils import set_global_cuda_envvars
from algorithms.utils.arguments import maybe_load_from_checkpoint, get_algo_class, parse_args
from utils.utils import log


def run(cfg):
    cfg = maybe_load_from_checkpoint(cfg)

    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()

    log.info('Exit...')
    return status


def main():
    """Script entry point."""
    set_global_cuda_envvars()

    cfg = parse_args()
    return run(cfg)


if __name__ == '__main__':
    sys.exit(main())
