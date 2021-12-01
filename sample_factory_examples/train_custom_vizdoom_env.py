"""
Example of how to use VizDoom env API to use your own custom VizDoom environment with Sample Factory.

To train:
python -m sample_factory_examples.train_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sample_factory_examples.enjoy_custom_vizdoom_env --algo=APPO --env=doom_my_custom_env --experiment=example

"""
import os
import sys
from os.path import join

from sample_factory.algorithms.utils.arguments import arg_parser, parse_args
from sample_factory.envs.doom.action_space import doom_action_space_extended
from sample_factory.envs.doom.doom_utils import DoomSpec, register_additional_doom_env
from sample_factory.run_algorithm import run_algorithm


def custom_parse_args(argv=None, evaluation=False):
    """
    Parse default SampleFactory arguments and add user-defined arguments on top.
    Allow to override argv for unit tests. Default value (None) means use sys.argv.
    Setting the evaluation flag to True adds additional CLI arguments for evaluating the policy (see the enjoy_ script).

    """
    parser = arg_parser(argv, evaluation=evaluation)

    # add custom args here
    parser.add_argument('--my_custom_doom_arg', type=int, default=300, help='Any custom arguments users might define')

    # SampleFactory parse_args function does some additional processing (see comments there)
    cfg = parse_args(argv=argv, evaluation=evaluation, parser=parser)
    return cfg


def register_custom_doom_env(custom_timeout):
    # absolute path needs to be specified, otherwise Doom will look in the SampleFactory scenarios folder
    scenario_absolute_path = join(os.path.dirname(__file__), 'doom_examples', 'custom_doom_env.cfg')
    spec = DoomSpec(
        'doom_my_custom_env',
        scenario_absolute_path,  # use your custom cfg here
        doom_action_space_extended(),
        reward_scaling=0.01, default_timeout=custom_timeout,
    )
    register_additional_doom_env(spec)


def main():
    """Script entry point."""
    cfg = custom_parse_args()
    register_custom_doom_env(custom_timeout=cfg.my_custom_doom_arg)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
