import math

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from sample_factory.algorithms.utils.action_distributions import calc_num_logits, get_action_distribution, is_continuous_action_space
from sample_factory.algorithms.utils.algo_utils import EPS
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.utils.utils import AttrDict
from sample_factory.utils.utils import log


# register custom encoders
ENCODER_REGISTRY = dict()


def register_custom_encoder(custom_encoder_name, encoder_cls):
    if custom_encoder_name in ENCODER_REGISTRY:
        log.warning('Encoder %s already registered', custom_encoder_name)

    assert issubclass(encoder_cls, EncoderBase), 'Custom encoders must be derived from EncoderBase'

    log.debug('Adding model class %r to registry (with name %s)', encoder_cls, custom_encoder_name)
    ENCODER_REGISTRY[custom_encoder_name] = encoder_cls


def get_hidden_size(cfg):
    if cfg.use_rnn:
        size = cfg.hidden_size * cfg.rnn_num_layers
    else:
        size = 1

    if cfg.rnn_type == 'lstm':
        size *= 2

    if not cfg.actor_critic_share_weights:
        # actor and critic need separate states
        size *= 2

    return size


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
    with torch.no_grad():
        mean = cfg.obs_subtract_mean
        scale = cfg.obs_scale

        if obs_dict['obs'].dtype != torch.float:
            obs_dict['obs'] = obs_dict['obs'].float()

        if abs(mean) > EPS:
            obs_dict['obs'].sub_(mean)

        if abs(scale - 1.0) > EPS:
            obs_dict['obs'].mul_(1.0 / scale)


class EncoderBase(nn.Module):
    def __init__(self, cfg, timing):
        super().__init__()

        self.cfg = cfg
        self.timing = timing

        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def get_encoder_out_size(self):
        return self.encoder_out_size

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
    class ConvEncoderImpl(nn.Module):
        """
        After we parse all the configuration and figure out the exact architecture of the model,
        we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
        fusion).
        """
        def __init__(self, activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape):
            super(ConvEncoder.ConvEncoderImpl, self).__init__()
            conv_layers = []
            for layer in conv_filters:
                if layer == 'maxpool_2x2':
                    conv_layers.append(nn.MaxPool2d((2, 2)))
                elif isinstance(layer, (list, tuple)):
                    inp_ch, out_ch, filter_size, stride = layer
                    conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                    conv_layers.append(activation)
                else:
                    raise NotImplementedError(f'Layer {layer} not supported!')

            self.conv_head = nn.Sequential(*conv_layers)
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            fc_layers = []
            for i in range(encoder_extra_fc_layers):
                size = self.conv_head_out_size if i == 0 else fc_layer_size
                fc_layers.extend([nn.Linear(size, fc_layer_size), activation])

            self.fc_layers = nn.Sequential(*fc_layers)

        def forward(self, obs):
            x = self.conv_head(obs)
            x = x.contiguous().view(-1, self.conv_head_out_size)
            x = self.fc_layers(x)
            return x

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

        activation = nonlinearity(self.cfg)
        fc_layer_size = fc_after_encoder_size(self.cfg)
        encoder_extra_fc_layers = self.cfg.encoder_extra_fc_layers

        enc = self.ConvEncoderImpl(activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, obs_dict):
        return self.enc(obs_dict['obs'])


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
                out = out + identity
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
        x = self.conv_head(obs_dict['obs'])
        x = x.contiguous().view(-1, self.conv_head_out_size)

        x = self.forward_fc_blocks(x)
        return x


class MlpEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        if cfg.encoder_subtype == 'mlp_mujoco':
            fc_encoder_layer = cfg.hidden_size
            encoder_layers = [
                nn.Linear(obs_shape.obs[0], fc_encoder_layer),
                nonlinearity(cfg),
                nn.Linear(fc_encoder_layer, fc_encoder_layer),
                nonlinearity(cfg),
            ]
        else:
            raise NotImplementedError(f'Unknown mlp encoder {cfg.encoder_subtype}')

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.init_fc_blocks(fc_encoder_layer)

    def forward(self, obs_dict):
        x = self.mlp_head(obs_dict['obs'])
        x = self.forward_fc_blocks(x)
        return x

def fc_layer(in_features, out_features, bias=True, spec_norm=False):
    if spec_norm:
        return spectral_norm(nn.Linear(in_features, out_features, bias))
    return nn.Linear(in_features, out_features, bias)

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


class PolicyCoreBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.core_output_size = -1

    def get_core_out_size(self):
        return self.core_output_size


class PolicyCoreRNN(PolicyCoreBase):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == 'gru':
            self.core = nn.GRU(input_size, cfg.hidden_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == 'lstm':
            self.core = nn.LSTM(input_size, cfg.hidden_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f'Unknown RNN type {cfg.rnn_type}')

        self.core_output_size = cfg.hidden_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.hidden_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states


class PolicyCoreFeedForward(PolicyCoreBase):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    def forward(self, head_output, fake_rnn_states):
        return head_output, fake_rnn_states


def create_core(cfg, core_input_size):
    if cfg.use_rnn:
        core = PolicyCoreRNN(cfg, core_input_size)
    else:
        core = PolicyCoreFeedForward(cfg, core_input_size)

    return core


class ActionsParameterizationBase(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space


class ActionParameterizationDefault(ActionsParameterizationBase):
    """
    A single fully-connected layer to output all parameters of the action distribution. Suitable for
    categorical action distributions, as well as continuous actions with learned state-dependent stddev.

    """

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        num_action_outputs = calc_num_logits(action_space)
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)

    def forward(self, actor_core_output):
        """Just forward the FC layer and generate the distribution object."""
        action_distribution_params = self.distribution_linear(actor_core_output)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution


class ActionParameterizationContinuousNonAdaptiveStddev(ActionsParameterizationBase):
    """Use a single learned parameter for action stddevs."""

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        assert not cfg.adaptive_stddev
        assert is_continuous_action_space(self.action_space), \
            'Non-adaptive stddev makes sense only for continuous action spaces'

        num_action_outputs = calc_num_logits(action_space)

        # calculate only action means using the policy neural network
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs // 2)

        # stddev is a single learned parameter
        initial_stddev = torch.empty([num_action_outputs // 2])
        initial_stddev.fill_(math.log(self.cfg.initial_stddev))
        self.learned_stddev = nn.Parameter(initial_stddev, requires_grad=True)

    def forward(self, actor_core_output):
        action_means = self.distribution_linear(actor_core_output)

        batch_size = action_means.shape[0]
        action_stddevs = self.learned_stddev.repeat(batch_size, 1)
        action_distribution_params = torch.cat((action_means, action_stddevs), dim=1)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution
