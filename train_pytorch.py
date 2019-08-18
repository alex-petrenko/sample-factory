import sys

from algorithms.utils.arguments import parse_args, get_algo_class
from envs.create_env import create_env


def train(cfg):
    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    agent = get_algo_class(cfg.algo)(make_env_func, cfg)
    agent.initialize()
    status = agent.learn()
    agent.finalize()
    return status


def main():
    """Script entry point."""
    cfg = parse_args()
    return train(cfg)


if __name__ == '__main__':
    sys.exit(main())
