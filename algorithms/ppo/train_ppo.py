import sys

from algorithms.ppo.agent_ppo import AgentPPO
from algorithms.utils.arguments import parse_args
from envs.doom.doom_utils import make_doom_env
from utils.utils import AttrDict


def train(args, ppo_params):
    def make_env_func(env_config):
        memento_args = AttrDict(dict(
            memento_size=ppo_params.memento_size,
            memento_increment=ppo_params.memento_increment,
            memento_history=ppo_params.memento_history,
        ))

        return make_doom_env(
            args.env, skip_frames=args.env_frameskip, pixel_format=args.pixel_format,
            memento_args=memento_args,
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
