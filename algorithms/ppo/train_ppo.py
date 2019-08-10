import sys

from algorithms.memento.memento_wrapper import get_memento_args
from algorithms.ppo.agent_ppo import AgentPPO
from algorithms.utils.arguments import parse_args
from envs.create_env import create_env


def train(args, ppo_params):
    def make_env_func(env_config):
        return create_env(
            args.env, skip_frames=args.env_frameskip, pixel_format=args.pixel_format,
            memento_args=get_memento_args(ppo_params),
            env_config=env_config,
        )

    agent = AgentPPO(make_env_func, params=ppo_params)
    agent.initialize()
    status = agent.learn()
    agent.finalize()
    return status


def main():
    """Script entry point."""
    args, params = parse_args(AgentPPO.Params)
    return train(args, params)


if __name__ == '__main__':
    sys.exit(main())
