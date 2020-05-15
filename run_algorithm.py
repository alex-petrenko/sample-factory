import sys

from algorithms.appo.appo_utils import set_global_cuda_envvars
from algorithms.utils.arguments import maybe_load_from_checkpoint, get_algo_class, parse_args


def run_algorithm(cfg):
    set_global_cuda_envvars()

    cfg = maybe_load_from_checkpoint(cfg)
    algo = get_algo_class(cfg.algo)(cfg)
    algo.initialize()
    status = algo.run()
    algo.finalize()
    return status


def main():
    """Script entry point."""
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
