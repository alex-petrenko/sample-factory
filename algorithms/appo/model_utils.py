import torch
from torch import nn

from algorithms.utils.algo_utils import EPS
from algorithms.utils.pytorch_utils import calc_num_elements
from utils.utils import AttrDict
from utils.utils import log


# register custom encoders
ENCODER_REGISTRY = dict()


def get_hidden_size(cfg):
    if not cfg.use_rnn:
        return 1

    if cfg.rnn_type == 'lstm':
        return cfg.hidden_size * 2
    else:
        return cfg.hidden_size


def fc_after_encoder_size(cfg):
    return cfg.hidden_size  # make configurable?


def nonlinearity(cfg):
    if cfg.nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif cfg.nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif cfg.nonlinearity == 'tanh':
        return nn.Tanh()
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


class EncoderBase(nn.Module):
    def __init__(self, cfg, timing):
        super().__init__()

        self.cfg = cfg
        self.timing = timing

        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def init_fc_blocks(self, input_size):
        layers = []
        fc_layer_size = fc_after_encoder_size(self.cfg)

        for i in range(self.cfg.encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size

            layers.extend([
                nn.Linear(size, fc_layer_size),
                nonlinearity(self.cfg),
            ])

        if len(layers) > 0:
            self.fc_after_enc = nn.Sequential(*layers)
            self.encoder_out_size = fc_layer_size
        else:
            self.encoder_out_size = input_size

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        """Default implementation, can be overridden in derived classes."""
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def forward_fc_blocks(self, x):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x)

        return x


class ConvEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

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
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)

        x = self.conv_head(obs_dict['obs'])
        x = x.view(-1, self.conv_head_out_size)

        x = self.forward_fc_blocks(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch, timing):
        super().__init__()

        self.timing = timing

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        with self.timing.add_time('res_block'):
            identity = x
            out = self.res_block_core(x)
            with self.timing.add_time('res_block_plus'):
                out.add_(identity)
            return out


class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.encoder_subtype == 'resnet_impala':
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels, self.timing))

            curr_input_channels = out_channels

        layers.append(nonlinearity(cfg))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        x = self.conv_head(obs_dict['obs'])
        x = x.view(-1, self.conv_head_out_size)

        x = self.forward_fc_blocks(x)
        return x


class MlpEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        if cfg.encoder_subtype == 'mlp_quads':
            fc_encoder_output_size = 256
            encoder_layers = [
                nn.Linear(obs_shape.obs[0], fc_encoder_output_size),
                nonlinearity(cfg),
            ]
        else:
            raise NotImplementedError(f'Unknown mlp encoder {cfg.encoder_subtype}')

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.init_fc_blocks(fc_encoder_output_size)

    def forward(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        x = self.mlp_head(obs_dict['obs'])

        x = self.forward_fc_blocks(x)
        return x


def create_encoder(cfg, obs_space, timing):
    if cfg.encoder_custom:
        encoder_cls = ENCODER_REGISTRY[cfg.encoder_custom]
        encoder = encoder_cls(cfg, obs_space, timing)
    else:
        encoder = create_standard_encoder(cfg, obs_space, timing)

    return encoder


def create_standard_encoder(cfg, obs_space, timing):
    if cfg.encoder_type == 'conv':
        encoder = ConvEncoder(cfg, obs_space, timing)
    elif cfg.encoder_type == 'resnet':
        encoder = ResnetEncoder(cfg, obs_space, timing)
    elif cfg.encoder_type == 'mlp':
        encoder = MlpEncoder(cfg, obs_space, timing)
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

        self.cfg = cfg

        self.core = nn.Sequential(
            nn.Linear(input_size, cfg.hidden_size),
            nonlinearity(cfg),
        )

    def forward(self, head_output, unused_rnn_states):
        x = self.core(head_output)
        fake_new_rnn_states = torch.zeros([x.shape[0], get_hidden_size(self.cfg)], device=x.device)
        return x, fake_new_rnn_states


def create_core(cfg, core_input_size):
    if cfg.use_rnn:
        core = PolicyCoreRNN(cfg, core_input_size)
    else:
        core = PolicyCoreFC(cfg, core_input_size)

    return core

