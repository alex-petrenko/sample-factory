"""
From the root of Sample Factory repo this can be run as:
python -m sample_factory_examples.train_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sample_factory_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example

"""
import random
import sys

import gym
import numpy as np

from sample_factory.algorithms.appo.model_utils import register_custom_encoder
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from sample_factory_examples.train_custom_env_custom_model import CustomEncoder, override_default_params_func


class CustomMultiEnv(gym.Env):
    """
    Implements a simple 2-agent game. Observation space is irrelevant. Optimal strategy is for both agents
    to choose the same action (both 0 or 1).

    """

    def __init__(self, full_env_name, cfg):
        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 8  # 8x8 images
        self.channels = 1  # it's easier when the channel dimension is present, even if it's 1

        self.observation_space = gym.spaces.Box(0, 1, (self.channels, self.res, self.res))
        self.action_space = gym.spaces.Discrete(2)

        self.num_agents = 2
        self.is_multiagent = True

        self.inactive_steps = [0] * self.num_agents

    def _obs(self):
        return [np.float32(np.random.rand(self.channels, self.res, self.res)) for _ in range(self.num_agents)]

    def reset(self):
        self.curr_episode_steps = 0
        return self._obs()

    def step(self, actions):
        infos = [dict() for _ in range(self.num_agents)]

        # random actions for inactive agents
        for agent_idx in range(self.num_agents):
            if self.inactive_steps[agent_idx] > 0:
                actions[agent_idx] = random.randint(0, 1)

        # "deactivate" agents randomly, mostly to test inactive agent masking functionality
        for agent_idx in range(self.num_agents):
            if self.inactive_steps[agent_idx] > 0:
                self.inactive_steps[agent_idx] -= 1
            else:
                if random.random() < 0.005:
                    self.inactive_steps[agent_idx] = random.randint(1, 48)

            infos[agent_idx]['is_active'] = self.inactive_steps[agent_idx] <= 0

        self.curr_episode_steps += 1

        payout_matrix = [
            [(-0.1, -0.1), (-0.2, -0.2)],
            [(-0.2, -0.25), (-0.1, -0.1)],  # make it asymmetric for easy learning, this is only a test after all
        ]

        # action = 0 to stay silent, 1 to betray
        rewards = payout_matrix[actions[0]][actions[1]]

        done = self.curr_episode_steps >= self.cfg.custom_env_episode_len
        dones = [done] * self.num_agents

        if done:
            # multi-agent environments should auto-reset!
            obs = self.reset()
        else:
            obs = self._obs()

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        pass


def make_custom_multi_env_func(full_env_name, cfg=None, env_config=None):
    return CustomMultiEnv(full_env_name, cfg)


def add_extra_params_func(env, parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument('--custom_env_episode_len', default=10, type=int, help='Number of steps in the episode')


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='my_custom_multi_env_',
        make_env_func=make_custom_multi_env_func,
        add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('custom_env_encoder', CustomEncoder)


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args()
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
