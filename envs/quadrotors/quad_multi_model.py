import torch
from torch import nn
from torch.nn.utils import spectral_norm

from algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from algorithms.utils.pytorch_utils import calc_num_elements


class QuadMultiMeanEncoder(EncoderBase):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space, timing, self_obs_dim=18, neighbor_obs_dim=6, neighbor_hidden_size=32):
        super().__init__(cfg, timing)
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.use_spectral_norm = cfg.use_spectral_norm

        fc_encoder_layer = cfg.hidden_size
        # encode the current drone's observations
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg)
        )

        # encode the neighboring drone's observations
        self.neighbor_encoder = nn.Sequential(
            fc_layer(self.neighbor_obs_dim, self.neighbor_hidden_size, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(self.neighbor_hidden_size, self.neighbor_hidden_size, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg)
        )

        self.self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))
        self.neighbor_encoder_out_size = calc_num_elements(self.neighbor_encoder, (self.neighbor_obs_dim,))

        # Feed forward self obs and neighbor obs after concatenation
        self.feed_forward = fc_layer(self.encoder_out_size + self.neighbor_encoder_out_size, cfg.hidden_size, spec_norm=self.use_spectral_norm)

        self.init_fc_blocks(cfg.hidden_size)

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self, obs_neighbors = obs[:, :self.self_obs_dim], obs[:, self.self_obs_dim:]
        self_embed = self.self_encoder(obs_self)

        # relative xyz and vxyz for the entire minibatch (batch dimension is batch_size * num_neighbors)
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.neighbor_encoder(obs_neighbors)
        batch_size = obs_self.shape[0]
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)

        mean_embed = torch.mean(neighbor_embeds, dim=1)
        embeddings = torch.cat((self_embed, mean_embed), dim=1)
        out = self.feed_forward(embeddings)
        return out


def register_models():
    quad_custom_encoder_name = 'quad_multi_encoder_deepset'
    if quad_custom_encoder_name not in ENCODER_REGISTRY:
        register_custom_encoder(quad_custom_encoder_name, QuadMultiMeanEncoder)
