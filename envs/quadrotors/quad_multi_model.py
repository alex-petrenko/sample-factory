import torch
from torch import nn

from algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from algorithms.utils.pytorch_utils import calc_num_elements


class QuadMultiMeanEncoder(EncoderBase):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space, timing, self_obs_dim=18, neighbor_obs_dim=6, neighbor_hidden_size=32, obstacle_obs_dim=6, obstacle_hidden_size=32):
        super().__init__(cfg, timing)
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.use_spectral_norm = cfg.use_spectral_norm
        self.obstacle_mode = cfg.quads_obstacle_mode
        self.num_use_neighbor_obs = int(cfg.quads_ratio_use_neighbor_obs * (cfg.quads_num_agents - 1))

        fc_encoder_layer = cfg.hidden_size
        # encode the current drone's observations
        self.self_encoder = nn.Sequential(
            fc_layer(self.self_obs_dim, fc_encoder_layer, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(fc_encoder_layer, fc_encoder_layer, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg)
        )
        self_encoder_out_size = calc_num_elements(self.self_encoder, (self.self_obs_dim,))

        # encode the neighboring drone's observations
        neighbor_encoder_out_size = 0
        if self.num_use_neighbor_obs > 0:
            self.neighbor_encoder = nn.Sequential(
                fc_layer(self.neighbor_obs_dim, self.neighbor_hidden_size, spec_norm=self.use_spectral_norm),
                nonlinearity(cfg),
                fc_layer(self.neighbor_hidden_size, self.neighbor_hidden_size, spec_norm=self.use_spectral_norm),
                nonlinearity(cfg)
            )
            neighbor_encoder_out_size = calc_num_elements(self.neighbor_encoder, (self.neighbor_obs_dim,))

        # encode the obstacle observations
        obstacle_encoder_out_size = 0
        if self.obstacle_mode != 'no_obstacles':
            self.obstacle_obs_dim = obstacle_obs_dim
            self.obstacle_hidden_size = obstacle_hidden_size
            self.obstacle_encoder = nn.Sequential(
                fc_layer(self.obstacle_obs_dim, self.obstacle_hidden_size, spec_norm=self.use_spectral_norm),
                nonlinearity(cfg),
                fc_layer(self.obstacle_hidden_size, self.obstacle_hidden_size, spec_norm=self.use_spectral_norm),
                nonlinearity(cfg),
            )
            obstacle_encoder_out_size = calc_num_elements(self.obstacle_encoder, (self.obstacle_obs_dim,))

        total_encoder_out_size = self_encoder_out_size + neighbor_encoder_out_size + obstacle_encoder_out_size

        # this is followed by another fully connected layer in the action parameterization, so we add a nonlinearity here
        self.feed_forward = nn.Sequential(
            fc_layer(total_encoder_out_size, cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nonlinearity(cfg),
        )

        self.encoder_out_size = cfg.hidden_size

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        self_embed = self.self_encoder(obs_self)
        embeddings = self_embed
        batch_size = obs_self.shape[0]
        # relative xyz and vxyz for the entire minibatch (batch dimension is batch_size * num_neighbors)
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        if self.num_use_neighbor_obs > 0:
            obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
            obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
            neighbor_embeds = self.neighbor_encoder(obs_neighbors)
            neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
            mean_embed = torch.mean(neighbor_embeds, dim=1)
            embeddings = torch.cat((embeddings, mean_embed), dim=1)

        if self.obstacle_mode != 'no_obstacles':
            obs_obstacles = obs[:, self.self_obs_dim + all_neighbor_obs_size:]
            obs_obstacles = obs_obstacles.reshape(-1, self.obstacle_obs_dim)
            obstacle_embeds = self.obstacle_encoder(obs_obstacles)
            obstacle_embeds = obstacle_embeds.reshape(batch_size, -1, self.obstacle_hidden_size)
            obstacle_mean_embed = torch.mean(obstacle_embeds, dim=1)
            embeddings = torch.cat((embeddings, obstacle_mean_embed), dim=1)

        out = self.feed_forward(embeddings)
        return out


def register_models():
    quad_custom_encoder_name = 'quad_multi_encoder_deepset'
    if quad_custom_encoder_name not in ENCODER_REGISTRY:
        register_custom_encoder(quad_custom_encoder_name, QuadMultiMeanEncoder)
