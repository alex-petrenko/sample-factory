import torch
from torch import nn

from algorithms.ppo.agent_ppo import calc_num_elements
from algorithms.utils.algo_utils import EPS
from utils.utils import AttrDict
from utils.utils import log


# register custom encoders
ENCODER_REGISTRY = dict()


def get_hidden_size(cfg):
    if cfg.rnn_type == 'lstm':
        return cfg.hidden_size * 2
    else:
        return cfg.hidden_size


def nonlinearity(cfg):
    if cfg.nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif cfg.nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown nonlinearity')


def get_obs_shape(obs_space):
    obs_shape = AttrDict()
    if hasattr(obs_space, 'spaces'):
        for key, space in obs_space.spaces.items():
            obs_shape[key] = space.shape
    else:
        obs_shape.obs = obs_space.shape

    return obs_shape


def normalize_obs(obs_dict, cfg):
    mean = cfg.obs_subtract_mean
    scale = cfg.obs_scale

    if obs_dict['obs'].dtype != torch.float:
        obs_dict['obs'] = obs_dict['obs'].float()

    if abs(mean) > EPS and abs(scale - 1.0) > EPS:
        obs_dict['obs'].sub_(mean).mul_(1.0 / scale)  # convert rgb observations to [-1, 1] in-place


class ConvEncoder(nn.Module):
    def __init__(self, cfg, obs_space):
        super().__init__()

        self.cfg = cfg

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.encoder_subtype == 'convnet_simple':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_subtype == 'convnet_impala':
            conv_filters = [[input_ch, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.encoder_subtype == 'minigrid_convnet_tiny':
            conv_filters = [[3, 16, 3, 1], [16, 32, 2, 1], [32, 64, 2, 1]]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.encoder_subtype}')

        conv_layers = []
        for layer in conv_filters:
            if layer == 'maxpool_2x2':
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(nonlinearity(cfg))
            else:
                raise NotImplementedError(f'Layer {layer} not supported!')

        self.conv_head = nn.Sequential(*conv_layers)
        self.encoder_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.encoder_out_size)

    def forward(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)

        x = self.conv_head(obs_dict['obs'])
        x = x.view(-1, self.encoder_out_size)

        return x


class MlpEncoder(nn.Module):
    def __init__(self, cfg, obs_space):
        super().__init__()

        self.cfg = cfg

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        if cfg.encoder_subtype == 'mlp_quads':
            fc_encoder_output_size = 256
            encoder_layers = [
                nn.Linear(obs_shape.obs[0], fc_encoder_output_size),
                nonlinearity(cfg),
            ]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.encoder_subtype}')

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.encoder_out_size = fc_encoder_output_size

    def forward(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        x = self.mlp_head(obs_dict['obs'])
        return x


def create_encoder(cfg, obs_space):
    if cfg.encoder_custom is not None:
        encoder_cls = ENCODER_REGISTRY[cfg.encoder_custom]
        encoder = encoder_cls(cfg, obs_space)
    else:
        encoder = create_standard_encoder(cfg, obs_space)

    return encoder


def create_standard_encoder(cfg, obs_space):
    if cfg.encoder_type == 'conv':
        encoder = ConvEncoder(cfg, obs_space)
    elif cfg.encoder_type == 'mlp':
        encoder = MlpEncoder(cfg, obs_space)
    else:
        raise Exception('Encoder type not supported')

    return encoder


class PolicyCoreRNN(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == 'gru':
            self.core = nn.GRUCell(input_size, cfg.hidden_size)
            self.is_gru = True
        elif cfg.rnn_type == 'lstm':
            self.core = nn.LSTMCell(input_size, cfg.hidden_size)
        else:
            raise RuntimeError(f'Unknown RNN type {cfg.rnn_type}')

    def forward(self, head_output, rnn_states):
        if self.is_gru:
            x = new_rnn_states = self.core(head_output, rnn_states)
        else:
            h, c = torch.split(rnn_states, self.cfg.hidden_size, dim=1)
            h, c = self.core(head_output, (h, c))
            x = h
            new_rnn_states = torch.cat((h, c), dim=1)

        return x, new_rnn_states


class PolicyCoreFC(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        self.core = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, head_output, unused_rnn_states):
        x = self.core(head_output)
        fake_new_rnn_states = torch.zeros(x.shape[0])  # optimize this away if you use the FC core
        return x, fake_new_rnn_states


def create_core(cfg, core_input_size):
    if cfg.use_rnn:
        core = PolicyCoreRNN(cfg, core_input_size)
    else:
        core = PolicyCoreFC(cfg, core_input_size)

    return core

