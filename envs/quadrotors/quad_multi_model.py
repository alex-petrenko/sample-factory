import torch
from torch import nn

from algorithms.appo.model_utils import nonlinearity, EncoderBase, \
    register_custom_encoder, ENCODER_REGISTRY, fc_layer
from algorithms.utils.pytorch_utils import calc_num_elements


class QuadNeighborhoodEncoder(nn.Module):
    def __init__(self, cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs):
        super().__init__()
        self.cfg = cfg
        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_dim = neighbor_obs_dim
        self.neighbor_hidden_size = neighbor_hidden_size
        self.num_use_neighbor_obs = num_use_neighbor_obs


class QuadNeighborhoodEncoderDeepsets(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.embedding_mlp = nn.Sequential(
            fc_layer(neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(batch_size, -1, self.neighbor_hidden_size)
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


class QuadNeighborhoodEncoderAttention(QuadNeighborhoodEncoder):
    def __init__(self, cfg, neighbor_obs_dim, neighbor_hidden_size, use_spectral_norm, self_obs_dim, num_use_neighbor_obs):
        super().__init__(cfg, self_obs_dim, neighbor_obs_dim, neighbor_hidden_size, num_use_neighbor_obs)

        self.self_obs_dim = self_obs_dim

        # outputs e_i from the paper
        self.embedding_mlp = nn.Sequential(
            fc_layer(self_obs_dim + neighbor_obs_dim, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg)
        )

        #  outputs h_i from the paper
        self.neighbor_value_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
        )

        # outputs scalar score alpha_i for each neighbor i
        self.attention_mlp = nn.Sequential(
            fc_layer(neighbor_hidden_size * 2, neighbor_hidden_size, spec_norm=use_spectral_norm),  # neighbor_hidden_size * 2 because we concat e_i and e_m
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, neighbor_hidden_size, spec_norm=use_spectral_norm),
            nonlinearity(cfg),
            fc_layer(neighbor_hidden_size, 1),
        )

    def forward(self, self_obs, obs, all_neighbor_obs_size, batch_size):
        obs_neighbors = obs[:, self.self_obs_dim:self.self_obs_dim + all_neighbor_obs_size]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # concatenate self observation with neighbor observation

        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        neighbor_embeddings = self.embedding_mlp(mlp_input)  # e_i in the paper https://arxiv.org/pdf/1809.08835.pdf

        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)  # h_i in the paper

        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(batch_size, -1, self.neighbor_hidden_size)
        neighbor_embeddings_mean = torch.mean(neighbor_embeddings_mean_input, dim=1)  # e_m in the paper
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(self.num_use_neighbor_obs, 1)

        attention_mlp_input = torch.cat((neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1)
        attention_weights = self.attention_mlp(attention_mlp_input).view(batch_size, -1)  # alpha_i in the paper
        attention_weights_softmax = torch.nn.functional.softmax(attention_weights, dim=1)
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)


        final_neighborhood_embedding = attention_weights_softmax * neighbor_values
        final_neighborhood_embedding = final_neighborhood_embedding.view(batch_size, -1, self.neighbor_hidden_size)
        final_neighborhood_embedding = torch.sum(final_neighborhood_embedding, dim=1)

        return final_neighborhood_embedding


class QuadMultiMeanEncoder(EncoderBase):
    # Mean embedding encoder based on the DeepRL for Swarms Paper
    def __init__(self, cfg, obs_space, timing, self_obs_dim=18, neighbor_obs_dim=6, neighbor_hidden_size=128, obstacle_obs_dim=6, obstacle_hidden_size=32):
        super().__init__(cfg, timing)
        self.neighbor_encoder_type = 'attention'  # TODO: config

        self.self_obs_dim = self_obs_dim
        self.neighbor_obs_type = cfg.neighbor_obs_type
        self.neighbor_hidden_size = neighbor_hidden_size
        self.use_spectral_norm = cfg.use_spectral_norm
        self.obstacle_mode = cfg.quads_obstacle_mode
        if cfg.quads_local_obs == -1:
            self.num_use_neighbor_obs = cfg.quads_num_agents - 1
        else:
            self.num_use_neighbor_obs = cfg.quads_local_obs

        if self.neighbor_obs_type == 'pos_vel_goals':
            self.neighbor_obs_dim = 9  # include goal pos info
        elif self.neighbor_obs_type == 'pos_vel':
            self.neighbor_obs_dim = neighbor_obs_dim
        elif self.neighbor_obs_type == 'attn':
            self.neighbor_obs_dim = 11
        elif cfg.neighbor_obs_type == 'none':
            # override these params so that neighbor encoder is a no-op during inference
            self.neighbor_obs_dim = 0
            self.num_use_neighbor_obs = 0
        else:
            raise NotImplementedError(f'Unknown value {cfg.neighbor_obs_type} passed to --neighbor_obs_type')

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
            if self.neighbor_encoder_type == 'deepsets':
                self.neighbor_encoder = QuadNeighborhoodEncoderDeepsets(cfg, self.neighbor_obs_dim, self.neighbor_hidden_size, self.use_spectral_norm, self.self_obs_dim, self.num_use_neighbor_obs)
            elif self.neighbor_encoder_type == 'attention':
                self.neighbor_encoder = QuadNeighborhoodEncoderAttention(cfg, self.neighbor_obs_dim, self.neighbor_hidden_size, self.use_spectral_norm, self.self_obs_dim, self.num_use_neighbor_obs)
            else:
                raise NotImplementedError()

            neighbor_encoder_out_size = self.neighbor_hidden_size

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
            fc_layer(total_encoder_out_size, 2 * cfg.hidden_size, spec_norm=self.use_spectral_norm),
            nn.Tanh(),
        )

        self.encoder_out_size = 2 * cfg.hidden_size

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        obs_self = obs[:, :self.self_obs_dim]
        self_embed = self.self_encoder(obs_self)
        embeddings = self_embed
        # embeddings = obs_self
        batch_size = obs_self.shape[0]
        # relative xyz and vxyz for the entire minibatch (batch dimension is batch_size * num_neighbors)
        all_neighbor_obs_size = self.neighbor_obs_dim * self.num_use_neighbor_obs
        if self.num_use_neighbor_obs > 0:
            neighborhood_embedding = self.neighbor_encoder(obs_self, obs, all_neighbor_obs_size, batch_size)
            embeddings = torch.cat((embeddings, neighborhood_embedding), dim=1)

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
