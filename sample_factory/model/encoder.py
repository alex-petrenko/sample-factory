from typing import Dict, List, Optional

import torch
from gym import spaces
from torch import Tensor, nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, get_obs_shape, model_device, nonlinearity
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(ModelModule):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class MlpEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        mlp_layers: List[int] = cfg.encoder_mlp_layers
        self.mlp_head = create_mlp(mlp_layers, obs_shape.obs[0], nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp_head = torch.jit.script(self.mlp_head)
        self.encoder_out_size = calc_num_elements(self.mlp_head, obs_shape.obs)

    def forward(self, obs_dict):
        x = self.mlp_head(obs_dict["obs"])
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ConvEncoderImpl(nn.Module):
    """
    After we parse all the configuration and figure out the exact architecture of the model,
    we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
    fusion).
    """

    def __init__(self, obs_shape: AttrDict, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module):
        super().__init__()

        conv_layers = []
        for layer in conv_filters:
            if layer == "maxpool_2x2":
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(activation)
            else:
                raise NotImplementedError(f"Layer {layer} not supported!")

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        self.mlp_layers = create_mlp(extra_mlp_layers, self.conv_head_out_size, activation)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


class ConvEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        obs_shape = get_obs_shape(obs_space)
        input_channels = obs_shape.obs[0]
        log.debug(f"{ConvEncoder.__name__}: {input_channels=}")

        if cfg.encoder_conv_architecture == "convnet_simple":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_conv_architecture == "convnet_impala":
            conv_filters = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.encoder_conv_architecture == "convnet_atari":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(f"Unknown encoder architecture {cfg.encoder_conv_architecture}")

        activation = nonlinearity(self.cfg)
        extra_mlp_layers: List[int] = cfg.encoder_conv_mlp_layers
        enc = ConvEncoderImpl(obs_shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug(f"Conv encoder output size: {self.encoder_out_size}")

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs_dict: Dict) -> Tensor:
        return self.enc(obs_dict["obs"])


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug("Num input channels: %d", input_ch)

        if cfg.encoder_conv_architecture == "resnet_impala":
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(cfg)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug(f"Convolutional layer output size: {self.conv_head_out_size}")

        self.mlp_layers = create_mlp(cfg.encoder_conv_mlp_layers, self.conv_head_out_size, activation)

        # should we do torch.jit here?

        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs_dict):
        x = self.conv_head(obs_dict["obs"])
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_img_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Make (most likely convolutional) encoder for image-based observations."""
    if cfg.encoder_conv_architecture.startswith("convnet"):
        return ConvEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("resnet"):
        return ResnetEncoder(cfg, obs_space)
    else:
        raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_architecture}")


def default_make_encoder_func(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """
    Analyze the observation space and create either a convolutional or an MLP encoder depending on
    whether this is an image-based environment or environment with vector observations.
    """

    # we only support dict observation spaces - envs with non-dict obs spaces use a wrapper
    # main subspace used to determine the encoder type is called "obs". For envs with multiple subspaces,
    # this function needs to be overridden (see vizdoom or dmlab encoders for example)
    assert isinstance(obs_space, spaces.Dict), f"Only dict observation spaces are supported, got {obs_space}"
    main_obs_space = obs_space.spaces["obs"]

    if isinstance(main_obs_space, spaces.Box):
        # determine whether this is an image-based environment
        ndim: int = len(main_obs_space.shape)
        if ndim == 1:
            return MlpEncoder(cfg, obs_space)
        elif 2 <= ndim <= 3:
            return make_img_encoder(cfg, obs_space)
        else:
            raise NotImplementedError(f"Unsupported observation space {main_obs_space}, {ndim=}")
    else:
        raise NotImplementedError(
            f"Observation space {main_obs_space} not supported. Override make encoder func or add support for it"
            f"in {__file__}"
        )
