import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.scaled import BottomLinesEncoder, TopLineEncoder


class SimBaConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()

        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.layer_norm = nn.GroupNorm(1, in_channels)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.layer_norm(x)
        out = self.conv_block(out)
        return residual + out


class SimBaCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Initial convolution to project to hidden dimension
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False)

        # SimBa residual blocks
        self.blocks = nn.ModuleList([SimBaConvBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])

        # Post-layer normalization
        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.post_norm = nn.GroupNorm(1, hidden_dim)

        # Global average pooling and final classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Initial projection
        x = self.initial_conv(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Post normalization
        x = self.post_norm(x)

        # Global pooling and classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def conv_outdim(i_dim, k, padding=0, stride=1, dilation=1):
    """Return the dimension after applying a convolution along one axis"""
    return int(1 + (i_dim + 2 * padding - dilation * (k - 1) - 1) / stride)


class SimBaEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_dim,
        depth,
        use_prev_action: bool = True,
    ):
        super().__init__()
        self.use_prev_action = use_prev_action

        in_channels = 2

        self.screen_encoder = SimBaCNN(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_blocks=depth,
        )
        self.topline_encoder = TopLineEncoder(hidden_dim)
        self.bottomline_encoder = BottomLinesEncoder(hidden_dim)

        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0

        screen_shape = obs_space["screen_image"].shape
        topline_shape = (obs_space["tty_chars"].shape[1],)
        bottomline_shape = (2 * obs_space["tty_chars"].shape[1],)
        self.out_dim = sum(
            [
                calc_num_elements(self.screen_encoder, screen_shape),
                calc_num_elements(self.topline_encoder, topline_shape),
                calc_num_elements(self.bottomline_encoder, bottomline_shape),
                self.prev_actions_dim,
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs_dict):
        B, C, H, W = obs_dict["screen_image"].shape

        topline = obs_dict["tty_chars"][..., 0, :]
        bottom_line = obs_dict["tty_chars"][..., -2:, :]

        encodings = [
            self.topline_encoder(topline.float(memory_format=torch.contiguous_format).view(B, -1)),
            self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(B, -1)),
            self.screen_encoder(obs_dict["screen_image"].float(memory_format=torch.contiguous_format).view(B, C, H, W)),
        ]

        if self.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(torch.nn.functional.one_hot(prev_actions, self.num_actions))

        encodings = self.fc(torch.cat(encodings, dim=1))

        return encodings


class SimBaActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimBaEncoder(
            obs_space=obs_space,
            hidden_dim=self.cfg.actor_hidden_dim,
            depth=self.cfg.actor_depth,
            use_prev_action=self.cfg.use_prev_action,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.cfg.actor_hidden_dim


class SimBaCriticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimBaEncoder(
            obs_space=obs_space,
            hidden_dim=self.cfg.critic_hidden_dim,
            depth=self.cfg.critic_depth,
            use_prev_action=self.cfg.use_prev_action,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.cfg.critic_hidden_dim


if __name__ == "__main__":
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.utils.attr_dict import AttrDict
    from sf_examples.nethack.train_nethack import parse_nethack_args, register_nethack_components

    register_nethack_components()
    cfg = parse_nethack_args(argv=["--env=nethack_score", "--add_image_observation=True"])

    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    obs, info = env.reset()
    encoder = SimBaCriticEncoder(cfg, env_info.obs_space)
    print(encoder)
    x = encoder(obs)
    print(x.shape)
