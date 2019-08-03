import argparse
import sys

from algorithms.utils.evaluation_config import add_eval_args
from envs.env_config import add_env_args
from utils.utils import log


def parse_args(params_cls, argv=None, evaluation=False):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--experiment', type=str, default=None, required=True)
    parser.add_argument('--env', type=str, default=None, required=True)

    # params object args
    params_cls.add_cli_args(parser)

    add_env_args(parser)

    if evaluation:
        add_eval_args(parser)

    args = parser.parse_args(argv)

    if not args.env or not args.experiment:
        raise Exception('--env and --experiment must be specified!')

    experiment = args.experiment

    params = params_cls(experiment)
    params.set_command_line(sys.argv)
    params.update(args)

    log.info('Config:')
    for arg in vars(args):
        log.info('%s %r', arg, getattr(args, arg))

    return args, params

