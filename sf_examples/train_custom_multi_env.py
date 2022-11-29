"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example_multi --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_multi_env --algo=APPO --env=my_custom_multi_env_v1 --experiment=example_multi

"""
from __future__ import annotations

import random
import sys
from typing import Any, Dict, Optional

import gym
import numpy as np

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sf_examples.train_custom_env_custom_model import make_custom_encoder, override_default_params


class CustomMultiEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    """
    Implements a simple 2-agent game. Observation space is irrelevant. Optimal strategy is for both agents
    to choose the same action (both 0 or 1).

    """

    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)

        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 8  # 8x8 images
        self.channels = 1  # it's easier when the channel dimension is present, even if it's 1

        self.observation_space = gym.spaces.Box(0, 1, (self.channels, self.res, self.res))
        self.action_space = gym.spaces.Discrete(2)

        self.num_agents = 2
        self.is_multiagent = True

        self.inactive_steps = [3] * self.num_agents

        self.episode_rewards = [[] for _ in range(self.num_agents)]

        self.reward_shaping = [dict(rew=-1.0) for _ in range(self.num_agents)]

        self.obs = None

        self.render_mode = render_mode

    def _obs(self):
        if self.obs is None:
            self.obs = [np.float32(np.random.rand(self.channels, self.res, self.res)) for _ in range(self.num_agents)]
        return self.obs

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        # log.debug(f"Episode reward: {self.episode_rewards} sum_0: {sum(self.episode_rewards[0])} sum_1: {sum(self.episode_rewards[1])}")
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        return self._obs(), [dict() for _ in range(self.num_agents)]

    def step(self, actions):
        infos = [dict() for _ in range(self.num_agents)]

        # "deactivate" agents randomly, mostly to test inactive agent masking functionality
        for agent_idx in range(self.num_agents):
            if self.inactive_steps[agent_idx] > 0:
                self.inactive_steps[agent_idx] -= 1
            else:
                if random.random() < 0.005:
                    self.inactive_steps[agent_idx] = random.randint(1, 48)

            infos[agent_idx]["is_active"] = self.inactive_steps[agent_idx] <= 0

        self.curr_episode_steps += 1

        # this is like prisoner's dilemma
        rew0 = self.reward_shaping[0]["rew"]
        rew1 = self.reward_shaping[1]["rew"]
        payout_matrix = [
            [(0, 0), (rew0, rew1)],
            [(rew0, rew1), (0, 0)],
        ]

        # action = 0 to stay silent, 1 to betray
        rewards = list(payout_matrix[actions[0]][actions[1]])
        for agent_idx in range(self.num_agents):
            if not infos[agent_idx]["is_active"]:
                rewards[agent_idx] = 0
            self.episode_rewards[agent_idx].append(rewards[agent_idx])

        time_out = self.curr_episode_steps >= self.cfg.custom_env_episode_len
        for agent_idx in range(self.num_agents):
            infos[agent_idx]["time_outs"] = time_out

        terminated = truncated = [time_out] * self.num_agents

        if time_out:
            # multi-agent environments should auto-reset!
            obs, infos = self.reset()
        else:
            obs = self._obs()

        return obs, rewards, terminated, truncated, infos

    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        return self.reward_shaping[0]

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        if isinstance(agent_idx, int):
            agent_idx = slice(agent_idx, agent_idx + 1)
        for idx in range(agent_idx.start, agent_idx.stop):
            self.reward_shaping[idx] = reward_shaping

    def render(self):
        pass


def make_custom_multi_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return CustomMultiEnv(full_env_name, cfg, render_mode=render_mode)


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=10, type=int, help="Number of steps in the episode")


def register_custom_components():
    register_env("my_custom_multi_env_v1", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
    # second parsing pass yields the final configuration
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
