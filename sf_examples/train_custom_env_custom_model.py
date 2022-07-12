"""
From the root of Sample Factory repo this can be run as:
python -m sf_examples.train_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example --save_every_sec=5 --experiment_summaries_interval=10

After training for a desired period of time, evaluate the policy by running:
python -m sf_examples.enjoy_custom_env_custom_model --algo=APPO --env=my_custom_env_v1 --experiment=example

"""

import sys

import gym
import numpy as np
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.model_utils import EncoderBase, get_obs_shape, nonlinearity, register_custom_encoder
from sample_factory.train import run_rl


class CustomEnv(gym.Env):
    def __init__(self, full_env_name, cfg):
        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 10  # 10x10 images
        self.channels = 1  # it's easier when the channel dimension is present, even if it's 1

        self.observation_space = gym.spaces.Box(0, 1, (self.channels, self.res, self.res))
        self.action_space = gym.spaces.Discrete(self.cfg.custom_env_num_actions)

    def _obs(self):
        return np.float32(np.random.rand(self.channels, self.res, self.res))

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        return self._obs()

    def step(self, action):
        # action should be an int here
        assert isinstance(action, (int, np.int32, np.int64))
        reward = action * 0.01

        done = self.curr_episode_steps >= self.cfg.custom_env_episode_len

        self.curr_episode_steps += 1

        return self._obs(), reward, done, dict()

    def render(self, mode="human"):
        pass


def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    return CustomEnv(full_env_name, cfg)


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
        encoder_custom="custom_env_encoder",
        hidden_size=128,
    )


class CustomEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)

        conv_layers = [
            nn.Conv2d(1, 8, 3, stride=2),
            nonlinearity(cfg),
            nn.Conv2d(8, 16, 2, stride=1),
            nonlinearity(cfg),
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict["obs"]

        x = self.conv_head(main_obs)
        x = x.view(-1, self.conv_head_out_size)

        # forward pass through configurable fully connected blocks immediately after the encoder
        x = self.forward_fc_blocks(x)
        return x


def register_custom_components():
    register_env("my_custom_env_v1", make_custom_env_func)
    register_custom_encoder("custom_env_encoder", CustomEncoder)


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
