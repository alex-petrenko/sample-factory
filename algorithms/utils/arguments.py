import argparse
import sys

from algorithms.ppo.agent_ppo import AgentPPO
from algorithms.utils.agent import Agent
from algorithms.utils.evaluation_config import add_eval_args
from envs.env_config import add_env_args, env_override_defaults
from utils.utils import log


def get_algo_class(algo):
    algo_class = Agent

    if algo == 'PPO':
        algo_class = AgentPPO
    else:
        log.warning('Algorithm %s is not supported', algo)

    return algo_class


def parse_args(argv=None, evaluation=False):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--algo', type=str, default=None, required=True)
    parser.add_argument('--env', type=str, default=None, required=True)
    parser.add_argument('--experiment', type=str, default=None, required=True)
    parser.add_argument('--experiments_root', type=str, default=None, required=False, help='If not None, store experiment data in the specified subfolder of train_dir. Useful for groups of experiments (e.g. gridsearch)')

    basic_args, _ = parser.parse_known_args(argv)
    algo = basic_args.algo
    env = basic_args.env

    # algorithm-specific parameters (e.g. for PPO)
    algo_class = get_algo_class(algo)
    algo_class.add_cli_args(parser)

    # env-specific parameters (e.g. for Doom env)
    add_env_args(env, parser)

    # env-specific default values for algo parameters (e.g. model size and convolutional head configuration)
    env_override_defaults(env, parser)

    if evaluation:
        add_eval_args(parser)

    # parse all the arguments (algo, env, and optionally evaluation)
    args = parser.parse_args(argv)

    args.command_line = ' '.join(argv)

    return args


def default_cfg(algo='PPO', env='env', experiment='test'):
    """Useful for tests."""
    return parse_args(argv=[f'--algo={algo}', f'--env={env}', f'--experiment={experiment}'])
