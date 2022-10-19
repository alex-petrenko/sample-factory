"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example

"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import gym
import numpy as np
from torch import nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, ObsSpace


# add "TrainingInfoInterface" and "RewardShapingInterface" just to demonstrate how to use them (and for testing)
class CustomEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)
        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 10  # 10x10 images
        self.channels = 1  # it's easier when the channel dimension is present, even if it's 1

        self.observation_space = gym.spaces.Box(0, 1, (self.channels, self.res, self.res))
        self.action_space = gym.spaces.Discrete(self.cfg.custom_env_num_actions)

        self.reward_shaping: Dict[str, Any] = dict(action_rew_coeff=0.01)

        self.render_mode = render_mode

    def _obs(self):
        return np.float32(np.random.rand(self.channels, self.res, self.res))

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        return self._obs(), {}

    def step(self, action):
        # action should be an int here
        assert isinstance(action, (int, np.int32, np.int64))
        reward = action * self.reward_shaping["action_rew_coeff"]

        terminated = truncated = self.curr_episode_steps >= self.cfg.custom_env_episode_len

        self.curr_episode_steps += 1

        return self._obs(), reward, terminated, truncated, dict()

    def render(self):
        pass

    def get_default_reward_shaping(self) -> Dict[str, Any]:
        return self.reward_shaping

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        self.reward_shaping = reward_shaping


def make_custom_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return CustomEnv(full_env_name, cfg, render_mode=render_mode)


def add_extra_params(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_num_actions", default=10, type=int, help="Number of actions in my custom env")
    p.add_argument("--custom_env_episode_len", default=1000, type=int, help="Number of steps in the episode")


def override_default_params(parser):
    """
    Override default argument values for this family of environments.
    All experiments for environments from my_custom_env_ family will have these parameters unless
    different values are passed from command line.

    """
    parser.set_defaults(
        rnn_size=128,
    )


class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = obs_space["obs"].shape

        conv_layers = [
            nn.Conv2d(1, 8, 3, stride=2),
            nonlinearity(cfg),
            nn.Conv2d(8, 16, 2, stride=1),
            nonlinearity(cfg),
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict["obs"]

        x = self.conv_head(main_obs)
        x = x.view(-1, self.conv_head_out_size)
        return x

    def get_out_size(self) -> int:
        return self.conv_head_out_size


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return CustomEncoder(cfg, obs_space)


def register_custom_components():
    register_env("my_custom_env_v1", make_custom_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)


def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_extra_params(parser)
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
