import sys

import torch
import torch.nn as nn

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.actor_critic import (
    ActorCritic,
    ActorCriticSeparateWeights,
    ActorCriticSharedWeights,
    obs_space_without_action_mask,
)
from sample_factory.model.encoder import Encoder
from sample_factory.model.utils import orthogonal_init
from sample_factory.train import run_rl
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sf_examples.nethack.models import (
    ChaoticDwarvenGPT5,
    ScaledNet,
    SimBaActorEncoder,
    SimBaCriticEncoder,
    ViTActorEncoder,
    ViTCriticEncoder,
)
from sf_examples.nethack.nethack_env import NETHACK_ENVS, make_nethack_env
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_model,
    add_extra_params_nethack_env,
    add_extra_params_vit_model,
    nethack_override_defaults,
)


def register_nethack_envs():
    for env_name in NETHACK_ENVS.keys():
        register_env(env_name, make_nethack_env)


class ActorCriticDifferentEncoders(ActorCriticSeparateWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)

        self.actor_encoder = SimBaActorEncoder(cfg, obs_space)
        self.actor_core = model_factory.make_model_core_func(cfg, self.actor_encoder.get_out_size())

        self.critic_encoder = SimBaCriticEncoder(cfg, obs_space)
        self.critic_core = model_factory.make_model_core_func(cfg, self.critic_encoder.get_out_size())

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]

        self.core_func = self._core_rnn if self.cfg.use_rnn else self._core_empty

        self.actor_decoder = model_factory.make_model_decoder_func(cfg, self.actor_core.get_out_size())
        self.critic_decoder = model_factory.make_model_decoder_func(cfg, self.critic_core.get_out_size())
        self.decoders = [self.actor_decoder, self.critic_decoder]

        self.critic_linear = orthogonal_init(nn.Linear(self.critic_decoder.get_out_size(), 1), gain=1.0)
        self.action_parameterization = self.get_action_parameterization(self.actor_decoder.get_out_size())

        self.encoder_outputs_sizes = [encoder.get_out_size() for encoder in self.encoders]
        self.rnn_hidden_sizes = [core.core.hidden_size * 2 for core in self.cores]
        self.core_outputs_sizes = [decoder.get_out_size() for decoder in self.decoders]

        # initalize manually
        # self.apply(self.initialize_weights)


def make_nethack_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.context import global_model_factory

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)

    if cfg.model == "simba":
        if cfg.actor_critic_share_weights:
            return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            return ActorCriticDifferentEncoders(model_factory, obs_space, action_space, cfg)
    elif cfg.model in ["cnn_pixels", "cnn_embeddings", "vit"]:
        if cfg.actor_critic_share_weights:
            return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)
    else:
        raise NotImplementedError


def make_nethack_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    if cfg.model == "vit":
        return ViTActorEncoder(cfg, obs_space)
    if cfg.model == "simba":
        return SimBaActorEncoder(cfg, obs_space)
    elif cfg.model == "cnn_pixels":
        return ChaoticDwarvenGPT5(cfg, obs_space)
    elif cfg.model == "cnn_embeddings":
        return ScaledNet(cfg, obs_space)


def register_nethack_components():
    register_nethack_envs()
    global_model_factory().register_encoder_factory(make_nethack_encoder)
    global_model_factory().register_actor_critic_factory(make_nethack_actor_critic)


def parse_nethack_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_vit_model(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()
    cfg = parse_nethack_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
