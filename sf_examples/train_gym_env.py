"""
An example that shows how to use SampleFactory with a Gym env.

Example command line for CartPole-v1:
python -m sf_examples.train_gym_env --algo=APPO --use_rnn=False --num_envs_per_worker=20 --policy_workers_per_policy=2 --recurrence=1 --with_vtrace=False --batch_size=512 --reward_scale=0.1 --save_every_sec=10 --experiment_summaries_interval=10 --experiment=example_gym_cartpole-v1 --env=CartPole-v1
python -m sf_examples.enjoy_gym_env --algo=APPO --experiment=example_gym_cartpole-v1 --env=CartPole-v1

"""

import sys
from typing import Optional

import gym

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl


def make_gym_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    return gym.make(full_env_name, render_mode=render_mode)


def register_custom_components():
    register_env("CartPole-v1", make_gym_env_func)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
