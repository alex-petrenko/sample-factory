"""Adapted from Scaling Laws for Imitation Learning in NetHack:
https://arxiv.org/abs/2307.09423

Credit to Jens Tuyls
"""

import math
from typing import List

import torch
from nle import nethack  # noqa: E402
from nle.nethack.nethack import TERMINAL_SHAPE
from torch import nn
from torch.nn import functional as F

from sample_factory.model.encoder import Encoder
from sample_factory.model.utils import he_normal_init, orthogonal_init
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.crop import Crop
from sf_examples.nethack.models.utils import interleave

PAD_CHAR = 0
NUM_CHARS = 256


class ScaledNet(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        self.use_prev_action = cfg.use_prev_action
        self.msg_hdim = cfg.msg_hdim
        self.h_dim = cfg.h_dim
        self.il_mode = False
        self.scale_cnn_channels = 1
        self.num_lstm_layers = 1
        self.num_fc_layers = 2
        self.num_screen_fc_layers = 1
        self.color_edim = cfg.color_edim
        self.char_edim = cfg.char_edim
        self.crop_dim = 9
        self.crop_out_filters = 8
        self.crop_num_layers = 5
        self.crop_inter_filters = 16
        self.crop_padding = 1
        self.crop_kernel_size = 3
        self.crop_stride = 1
        self.use_crop = cfg.use_crop
        self.use_resnet = cfg.use_resnet
        self.use_crop_norm = cfg.use_crop_norm
        self.action_embedding_dim = 32
        self.obs_frame_stack = 1
        self.num_res_blocks = 2
        self.num_res_layers = 2
        self.screen_shape = TERMINAL_SHAPE
        self.screen_kernel_size = cfg.screen_kernel_size
        self.no_max_pool = cfg.no_max_pool
        self.screen_conv_blocks = cfg.screen_conv_blocks
        self.blstats_hdim = cfg.blstats_hdim if cfg.blstats_hdim else cfg.h_dim
        self.fc_after_cnn_hdim = cfg.fc_after_cnn_hdim if cfg.fc_after_cnn_hdim else cfg.h_dim

        # NOTE: -3 because we cut the topline and bottom two lines
        if self.use_crop:
            self.crop = Crop(self.screen_shape[0] - 3, self.screen_shape[1], self.crop_dim, self.crop_dim)
            crop_in_channels = [self.char_edim + self.color_edim] + [self.crop_inter_filters] * (
                self.crop_num_layers - 1
            )
            crop_out_channels = [self.crop_inter_filters] * (self.crop_num_layers - 1) + [self.crop_out_filters]
            conv_extract_crop = []
            norm_extract_crop = []
            for i in range(self.crop_num_layers):
                conv_extract_crop.append(
                    nn.Conv2d(
                        in_channels=crop_in_channels[i],
                        out_channels=crop_out_channels[i],
                        kernel_size=(self.crop_kernel_size, self.crop_kernel_size),
                        stride=self.crop_stride,
                        padding=self.crop_padding,
                    )
                )
                norm_extract_crop.append(nn.BatchNorm2d(crop_out_channels[i]))

            if self.use_crop_norm:
                self.extract_crop_representation = nn.Sequential(
                    *interleave(conv_extract_crop, norm_extract_crop, [nn.ELU()] * len(conv_extract_crop))
                )
            else:
                self.extract_crop_representation = nn.Sequential(
                    *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract_crop))
                )
            self.crop_out_dim = self.crop_dim**2 * self.crop_out_filters
        else:
            self.crop_out_dim = 0

        self.topline_encoder = TopLineEncoder(msg_hdim=self.msg_hdim)
        self.bottomline_encoder = BottomLinesEncoder(h_dim=self.blstats_hdim // 4)

        self.screen_encoder = CharColorEncoderResnet(
            (self.screen_shape[0] - 3, self.screen_shape[1]),
            h_dim=self.fc_after_cnn_hdim,
            num_fc_layers=self.num_screen_fc_layers,
            scale_cnn_channels=self.scale_cnn_channels,
            color_edim=self.color_edim,
            char_edim=self.char_edim,
            obs_frame_stack=self.obs_frame_stack,
            num_res_blocks=self.num_res_blocks,
            num_res_layers=self.num_res_layers,
            kernel_size=self.screen_kernel_size,
            no_max_pool=self.no_max_pool,
            screen_conv_blocks=self.screen_conv_blocks,
        )

        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0

        self.out_dim = sum(
            [
                self.topline_encoder.msg_hdim,
                self.bottomline_encoder.h_dim,
                self.screen_encoder.h_dim,
                self.prev_actions_dim,
                self.crop_out_dim,
            ]
        )

        fc_layers = [nn.Linear(self.out_dim, self.h_dim), nn.ReLU()]
        for _ in range(self.num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.h_dim, self.h_dim))
            fc_layers.append(nn.ReLU())
        self.fc = nn.Sequential(*fc_layers)

        self.encoder_out_size = self.h_dim

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, obs_dict):
        B, H, W = obs_dict["tty_chars"].shape
        # to process images with CNNs we need channels dim
        C = 1

        # Take last channel for now
        topline = obs_dict["tty_chars"][:, 0].contiguous()
        bottom_line = obs_dict["tty_chars"][:, -2:].contiguous()

        # Blstats
        blstats_rep = self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(B, -1))

        encodings = [
            self.topline_encoder(topline.float(memory_format=torch.contiguous_format).view(B, -1)),
            blstats_rep,
        ]

        # Main obs encoding
        tty_chars = (
            obs_dict["tty_chars"][:, 1:-2]
            .contiguous()
            .float(memory_format=torch.contiguous_format)
            .view(B, C, H - 3, W)
        )
        tty_colors = obs_dict["tty_colors"][:, 1:-2].contiguous().view(B, C, H - 3, W)
        tty_cursor = obs_dict["tty_cursor"].contiguous().view(B, -1)
        encodings.append(self.screen_encoder(tty_chars, tty_colors))

        # Previous action encoding
        if self.use_prev_action:
            encodings.append(torch.nn.functional.one_hot(obs_dict["prev_actions"].long(), self.num_actions).view(B, -1))

        # Crop encoding
        if self.use_crop:
            # very important! otherwise we'll mess with tty_cursor below
            # uint8 is needed for -1 operation to work properly 0 -> 255
            tty_cursor = tty_cursor.clone().to(torch.uint8)
            tty_cursor[:, 0] -= 1  # adjust y position for cropping below
            tty_cursor = tty_cursor.flip(-1)  # flip (y, x) to be (x, y)
            crop_tty_chars = self.crop(tty_chars[..., -1, :, :], tty_cursor)
            crop_tty_colors = self.crop(tty_colors[..., -1, :, :], tty_cursor)
            crop_chars = selectt(self.screen_encoder.char_embeddings, crop_tty_chars.long(), True)
            crop_colors = selectt(self.screen_encoder.color_embeddings, crop_tty_colors.long(), True)
            crop_obs = torch.cat([crop_chars, crop_colors], dim=-1)
            encodings.append(self.extract_crop_representation(crop_obs.permute(0, 3, 1, 2).contiguous()).view(B, -1))

        encodings = self.fc(torch.cat(encodings, dim=1))

        return encodings


class CharColorEncoderResnet(nn.Module):
    """
    Inspired by network from IMPALA https://arxiv.org/pdf/1802.01561.pdf
    """

    def __init__(
        self,
        screen_shape,
        h_dim: int = 512,
        scale_cnn_channels: int = 1,
        num_fc_layers: int = 1,
        char_edim: int = 16,
        color_edim: int = 16,
        obs_frame_stack: int = 1,
        num_res_blocks: int = 2,
        num_res_layers: int = 2,
        kernel_size: int = 3,
        no_max_pool: bool = False,
        screen_conv_blocks: int = 3,
    ):
        super(CharColorEncoderResnet, self).__init__()

        self.h, self.w = screen_shape
        self.h_dim = h_dim
        self.num_fc_layers = num_fc_layers
        self.char_edim = char_edim
        self.color_edim = color_edim
        self.no_max_pool = no_max_pool
        self.screen_conv_blocks = screen_conv_blocks

        self.blocks = []

        self.conv_params = [
            [
                char_edim * obs_frame_stack + color_edim * obs_frame_stack,
                int(16 * scale_cnn_channels),
                kernel_size,
                num_res_blocks,
            ],
            [int(16 * scale_cnn_channels), int(32 * scale_cnn_channels), kernel_size, num_res_blocks],
            [int(32 * scale_cnn_channels), int(32 * scale_cnn_channels), kernel_size, num_res_blocks],
        ]

        self.conv_params = self.conv_params[: self.screen_conv_blocks]

        for in_channels, out_channels, filter_size, num_res_blocks in self.conv_params:
            block = []
            # Downsample
            block.append(nn.Conv2d(in_channels, out_channels, filter_size, stride=1, padding=(filter_size // 2)))
            if not self.no_max_pool:
                block.append(nn.MaxPool2d(kernel_size=3, stride=2))
                self.h = math.floor((self.h - 1 * (3 - 1) - 1) / 2 + 1)  # from PyTorch Docs
                self.w = math.floor((self.w - 1 * (3 - 1) - 1) / 2 + 1)  # from PyTorch Docs

            # Residual block(s)
            for _ in range(num_res_blocks):
                block.append(ResBlock(out_channels, out_channels, filter_size, num_res_layers))
            self.blocks.append(nn.Sequential(*block))

        self.conv_net = nn.Sequential(*self.blocks)
        self.out_size = self.h * self.w * out_channels

        fc_layers = [nn.Linear(self.out_size, self.h_dim), nn.ELU(inplace=True)]
        for _ in range(self.num_fc_layers - 1):
            fc_layers.append(nn.Linear(self.h_dim, self.h_dim))
            fc_layers.append(nn.ELU(inplace=True))
        self.fc_head = nn.Sequential(*fc_layers)

        self.char_embeddings = nn.Embedding(256, self.char_edim)
        self.color_embeddings = nn.Embedding(128, self.color_edim)

    def forward(self, chars, colors):
        chars, colors = self._embed(chars, colors)  # 21 x 80
        x = self._stack(chars, colors)
        x = self.conv_net(x)
        x = x.view(-1, self.out_size)
        x = self.fc_head(x)
        return x

    def _embed(self, chars, colors):
        chars = selectt(self.char_embeddings, chars.long(), True)
        colors = selectt(self.color_embeddings, colors.long(), True)
        return chars, colors

    def _stack(self, chars, colors):
        obs = torch.cat([chars, colors], dim=-1)
        return obs.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filter_size: int, num_layers: int):
        super(ResBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, filter_size, stride=1, padding=(filter_size // 2)))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ELU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) + x


class BottomLinesEncoder(nn.Module):
    """
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """

    def __init__(self, h_dim: int = 128, scale_cnn_channels: int = 1):
        super(BottomLinesEncoder, self).__init__()
        self.conv_layers = []
        w = nethack.NLE_TERM_CO * 2
        for in_ch, out_ch, filter, stride in [
            [2, int(32 * scale_cnn_channels), 8, 4],
            [int(32 * scale_cnn_channels), int(64 * scale_cnn_channels), 4, 1],
        ]:
            self.conv_layers.append(orthogonal_init(nn.Conv1d(in_ch, out_ch, filter, stride=stride), gain=1.0))
            self.conv_layers.append(nn.ELU(inplace=True))
            w = conv_outdim(w, filter, padding=0, stride=stride)

        self.h_dim = h_dim

        self.out_dim = w * out_ch
        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fwd_net = nn.Sequential(
            orthogonal_init(nn.Linear(self.out_dim, self.h_dim), gain=1.0),
            nn.ELU(inplace=True),
            orthogonal_init(nn.Linear(self.h_dim, self.h_dim), gain=1.0),
            nn.ELU(inplace=True),
        )

    def forward(self, bottom_lines):
        B, D = bottom_lines.shape
        # ASCII 32: ' ', ASCII [33-128]: visible characters
        chars_normalised = (bottom_lines - 32) / 96

        # ASCII [45-57]: -./01234556789
        numbers_mask = (bottom_lines > 44) * (bottom_lines < 58)
        digits_normalised = numbers_mask * (bottom_lines - 47) / 10  # why subtract 47 here and not 48?

        # Put in different channels & conv (B, 2, D)
        x = torch.stack([chars_normalised, digits_normalised], dim=1)
        return self.fwd_net(self.conv_net(x).view(B, -1))


class TopLineEncoder(nn.Module):
    """
    This class uses a one-hot encoding of the ASCII characters
    as features that get fed into an MLP.
    Adapted from https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022/blob/67139262966aa11555cf7aca15723375b36fbe42/experiment_code/hackrl/models/offline_chaotic_dwarf.py
    """

    def __init__(self, msg_hdim: int):
        super(TopLineEncoder, self).__init__()
        self.msg_hdim = msg_hdim
        self.i_dim = nethack.NLE_TERM_CO * 256

        self.msg_fwd = nn.Sequential(
            orthogonal_init(nn.Linear(self.i_dim, self.msg_hdim), gain=1.0),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(self.msg_hdim, self.msg_hdim), gain=1.0),
            nn.ReLU(inplace=True),
        )

    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = F.one_hot((message).long(), 256).reshape(-1, self.i_dim).float()
        return self.msg_fwd(message_normed)


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


def selectt(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)
