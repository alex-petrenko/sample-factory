"""
An example that shows how to use SampleFactory with a PettingZoo env.

Example command line for tictactoe_v3:
python -m sf_examples.train_pettingzoo_env --algo=APPO --use_rnn=False --num_envs_per_worker=20 --policy_workers_per_policy=2 --recurrence=1 --with_vtrace=False --batch_size=512 --save_every_sec=10 --experiment_summaries_interval=10 --experiment=example_pettingzoo_tictactoe_v3 --env=tictactoe_v3
python -m sf_examples.enjoy_pettingzoo_env --algo=APPO --experiment=example_pettingzoo_tictactoe_v3 --env=tictactoe_v3

"""

import sys
from typing import List, Optional

import gymnasium as gym
import torch
from pettingzoo.classic import tictactoe_v3
from pettingzoo.utils import turn_based_aec_to_parallel
from torch import Tensor, nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.envs.pettingzoo_envs import PettingZooParallelEnv
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import create_mlp, nonlinearity
from sample_factory.train import run_rl
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace


class CustomConvEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        main_obs_space = obs_space["obs"]
        input_channels = main_obs_space.shape[0]
        conv_filters = [[input_channels, 32, 2, 1], [32, 64, 2, 1], [64, 128, 2, 1]]
        activation = nonlinearity(self.cfg)
        extra_mlp_layers = cfg.encoder_conv_mlp_layers
        enc = ConvEncoderImpl(main_obs_space.shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)
        self.encoder_out_size = calc_num_elements(self.enc, main_obs_space.shape)

    def get_out_size(self):
        return self.encoder_out_size

    def forward(self, obs_dict):
        main_obs = obs_dict["obs"]
        return self.enc(main_obs)


class ConvEncoderImpl(nn.Module):
    def __init__(self, obs_shape: AttrDict, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module):
        super().__init__()
        conv_layers = []

        for layer in conv_filters:
            inp_ch, out_ch, filter_size, padding = layer
            conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, padding=padding))
            conv_layers.append(activation)

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)
        self.mlp_layers = create_mlp(extra_mlp_layers, self.conv_head_out_size, activation)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


def make_pettingzoo_env(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    return PettingZooParallelEnv(turn_based_aec_to_parallel(tictactoe_v3.env(render_mode=render_mode)))


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return CustomConvEncoder(cfg, obs_space)


def register_custom_components():
    register_env("tictactoe_v3", make_pettingzoo_env)
    global_model_factory().register_encoder_factory(make_custom_encoder)


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
