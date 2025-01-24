"""Adapted from Chaos Dwarf in Nethack Challenge Starter Kit:
https://github.com/Miffyli/nle-sample-factory-baseline

MIT License

Copyright (c) 2021 Anssi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from nle import nethack
from torch import nn
from torch.nn import functional as F

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace


class MessageEncoder(nn.Module):
    def __init__(self):
        super(MessageEncoder, self).__init__()
        self.hidden_dim = 128
        self.msg_fwd = nn.Sequential(
            nn.Linear(nethack.MESSAGE_SHAPE[0], 128),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, message):
        return self.msg_fwd(message / 255.0)


class BLStatsEncoder(nn.Module):
    def __init__(self):
        super(BLStatsEncoder, self).__init__()
        self.hidden_dim = 128 + nethack.BLSTATS_SHAPE[0]
        self.blstats_fwd = nn.Sequential(
            nn.Linear(nethack.BLSTATS_SHAPE[0], 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
        )

        normalization_stats = torch.tensor(
            [
                1.0 / 79.0,  # hero col
                1.0 / 21,  # hero row
                0.0,  # strength pct
                1.0 / 10,  # strength
                1.0 / 10,  # dexterity
                1.0 / 10,  # constitution
                1.0 / 10,  # intelligence
                1.0 / 10,  # wisdom
                1.0 / 10,  # charisma
                0.0,  # score
                1.0 / 10,  # hitpoints
                1.0 / 10,  # max hitpoints
                0.0,  # depth
                1.0 / 1000,  # gold
                1.0 / 10,  # energy
                1.0 / 10,  # max energy
                1.0 / 10,  # armor class
                0.0,  # monster level
                1.0 / 10,  # experience level
                1.0 / 100,  # experience points
                1.0 / 1000,  # time
                1.0,  # hunger_state
                1.0 / 10,  # carrying capacity
                0.0,  # carrying capacity
                0.0,  # level number
                0.0,  # condition bits
                0.0,  # alignment bits
            ],
            requires_grad=False,
        )
        self.register_buffer("normalization_stats", normalization_stats)

        self.blstat_range = (-5, 5)

    def forward(self, blstats):
        norm_bls = torch.clip(
            blstats * self.normalization_stats,
            self.blstat_range[0],
            self.blstat_range[1],
        )

        return torch.cat([self.blstats_fwd(norm_bls), norm_bls], dim=-1)


class TopLineEncoder(nn.Module):
    def __init__(self):
        super(TopLineEncoder, self).__init__()
        self.hidden_dim = 128
        self.i_dim = nethack.NLE_TERM_CO * 256

        self.msg_fwd = nn.Sequential(
            nn.Linear(self.i_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, self.hidden_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, message):
        # Characters start at 33 in ASCII and go to 128. 96 = 128 - 32
        message_normed = F.one_hot((message).long(), 256).reshape(-1, self.i_dim).float()
        return self.msg_fwd(message_normed)


class BottomLinesEncoder(nn.Module):
    def __init__(self):
        super(BottomLinesEncoder, self).__init__()
        self.conv_layers = []
        w = nethack.NLE_TERM_CO * 2
        for in_ch, out_ch, filter, stride in [[2, 32, 8, 4], [32, 64, 4, 1]]:
            self.conv_layers.append(nn.Conv1d(in_ch, out_ch, filter, stride=stride))
            self.conv_layers.append(nn.ELU(inplace=True))
            w = conv_outdim(w, filter, padding=0, stride=stride)

        self.conv_net = nn.Sequential(*self.conv_layers)
        self.fwd_net = nn.Sequential(
            nn.Linear(w * out_ch, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
        )
        self.hidden_dim = 128

    def forward(self, bottom_lines):
        B, D = bottom_lines.shape
        # ASCII 32: ' ', ASCII [33-128]: visible characters
        chars_normalised = (bottom_lines - 32) / 96

        # ASCII [45-57]: -./01234556789
        numbers_mask = (bottom_lines > 44) * (bottom_lines < 58)
        digits_normalised = numbers_mask * (bottom_lines - 47) / 10

        # Put in different channels & conv (B, 2, D)
        x = torch.stack([chars_normalised, digits_normalised], dim=1)
        return self.fwd_net(self.conv_net(x).view(B, -1))


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


class InverseModel(nn.Module):
    def __init__(self, h_dim, action_space):
        super(InverseModel, self).__init__()
        self.h_dim = h_dim * 2
        self.action_space = action_space

        self.fwd_model = nn.Sequential(
            nn.Linear(self.h_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, action_space),
        )

    def forward(self, obs):
        T, B, *_ = obs.shape
        x = torch.cat([obs[:-1], obs[1:]], dim=-1)
        pred_a = self.fwd_model(x)
        off_by_one = torch.ones((1, B, self.action_space), device=x.device) * -1
        return torch.cat([pred_a, off_by_one], dim=0)


class CharColorEncoder(nn.Module):
    def __init__(
        self,
        screen_shape,
        char_edim: int = 16,
        color_edim: int = 16,
    ):
        super().__init__()
        conv_layers = []

        self.h, self.w = screen_shape
        self.char_edim = char_edim
        self.color_edim = color_edim
        self.hidden_dim = 512

        self.conv_filters = [
            [char_edim + color_edim, 32, (3, 5), (1, 2), (1, 2)],
            [32, 64, (3, 5), (1, 2), (1, 2)],
            [64, 128, 3, 1, 1],
            [128, 128, 3, 1, 1],
        ]

        for (
            in_channels,
            out_channels,
            filter_size,
            stride,
            dilation,
        ) in self.conv_filters:
            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    filter_size,
                    stride=stride,
                    dilation=dilation,
                )
            )
            conv_layers.append(nn.ELU(inplace=True))

        self.conv_head = nn.Sequential(*conv_layers)
        self.out_size = calc_num_elements(self.conv_head, (char_edim + color_edim,) + screen_shape)

        self.fc_head = nn.Sequential(nn.Linear(self.out_size, self.hidden_dim), nn.ELU(inplace=True))

        self.char_embeddings = nn.Embedding(256, self.char_edim)
        self.color_embeddings = nn.Embedding(128, self.color_edim)

    def forward(self, chars, colors):
        chars, colors = self._embed(chars, colors)  # 21 x 80
        x = self._stack(chars, colors)
        x = self.conv_head(x)
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


class ChaoticDwarvenGPT5(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        self.use_tty_only = cfg.use_tty_only
        self.use_prev_action = cfg.use_prev_action

        screen_shape = obs_space["tty_chars"].shape
        self.screen_encoder = CharColorEncoder(
            (screen_shape[0] - 3, screen_shape[1]),
            char_edim=cfg.char_edim,
            color_edim=cfg.color_edim,
        )

        # top and bottom encoders
        self.topline_encoder = TopLineEncoder()
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder())

        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0

        self.encoder_out_size = sum(
            [
                self.topline_encoder.hidden_dim,
                self.bottomline_encoder.hidden_dim,
                self.screen_encoder.hidden_dim,
                self.prev_actions_dim,
            ]
        )

    def get_out_size(self) -> int:
        return self.encoder_out_size

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
        encodings.append(self.screen_encoder(tty_chars, tty_colors))

        if self.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(torch.nn.functional.one_hot(prev_actions, self.num_actions))

        return torch.cat(encodings, dim=1)


def selectt(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        # Access weight through the embedding layer
        return nn.functional.embedding(x, embedding_layer.weight)
    else:
        # Use standard embedding forward
        return embedding_layer(x)


if __name__ == "__main__":
    # Test the screen encoder
    encoder = CharColorEncoder(
        (21 - 3, 80),
        char_edim=16,
        color_edim=16,
    )
    tty_chars = torch.zeros(160, 1, 21, 80)
    tty_colors = torch.zeros(160, 1, 21, 80)
    print(encoder(tty_chars, tty_colors).shape)
