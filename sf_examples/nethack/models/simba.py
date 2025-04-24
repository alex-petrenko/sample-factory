import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.utils import he_normal_init, orthogonal_init
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.scaled import BottomLinesEncoder, TopLineEncoder


class SimBaConvBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.layer_norm = nn.GroupNorm(1, in_channels)

        self.conv_block = nn.Sequential(
            he_normal_init(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)),
            nn.ELU(inplace=True),
            he_normal_init(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)),
        )

        # Add projection layer if channels change
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.layer_norm(x)
        out = self.conv_block(out)

        if self.projection is not None:
            identity = self.projection(identity)

        return identity + out


class SimBaCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=64,
        num_blocks=2,
        use_max_pool=False,
        expansion=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_max_pool = use_max_pool

        assert in_channels & (in_channels - 1) == 0, "in_channels must be power of 2"
        assert hidden_dim & (hidden_dim - 1) == 0, "hidden_dim must be power of 2"
        assert hidden_dim >= in_channels, "hidden_dim must be >= in_channels"
        assert not use_max_pool or (use_max_pool and num_blocks <= 4)

        # Calculate number of doublings needed
        current_channels = in_channels
        self.blocks = []

        # Initial convolution to project to hidden dimension
        self.initial_conv = orthogonal_init(
            nn.Conv2d(in_channels, current_channels * 2, kernel_size=3, padding=0, bias=False),
            gain=1.0,
        )
        current_channels *= 2

        # SimBa residual blocks
        self.blocks = []
        for i in range(num_blocks):
            next_channels = min(current_channels * 2, hidden_dim)
            self.blocks.append(SimBaConvBlock(current_channels, next_channels * expansion, next_channels))
            if self.use_max_pool:
                self.blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels
        self.blocks = nn.ModuleList(self.blocks)

        # Post-layer normalization
        # GroupNorm with num_groups=1 is equivalent to LayerNorm
        self.post_norm = nn.GroupNorm(1, current_channels)

        # Global average pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Initial projection
        x = self.initial_conv(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Post normalization
        x = self.post_norm(x)

        # Global pooling
        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        return x


class SimBaEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_dim,
        depth,
        use_prev_action: bool = True,
        use_learned_embeddings: bool = False,
        char_edim: int = 16,
        color_edim: int = 16,
        use_max_pool: bool = False,
        expansion: int = 2,
    ):
        super().__init__()
        self.use_prev_action = use_prev_action
        self.use_learned_embeddings = use_learned_embeddings
        self.char_edim = char_edim
        self.color_edim = color_edim

        self.char_embeddings = nn.Embedding(256, self.char_edim)
        self.color_embeddings = nn.Embedding(128, self.color_edim)
        C, W, H = obs_space["screen_image"].shape
        if self.use_learned_embeddings:
            in_channels = self.char_edim + self.color_edim
        else:
            in_channels = C

        self.screen_encoder = torch.jit.script(
            SimBaCNN(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_blocks=depth,
                use_max_pool=use_max_pool,
                expansion=expansion,
            )
        )
        self.topline_encoder = torch.jit.script(TopLineEncoder(hidden_dim))
        self.bottomline_encoder = torch.jit.script(BottomLinesEncoder(hidden_dim))

        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0

        screen_shape = (in_channels, W, H)
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
            orthogonal_init(nn.Linear(self.out_dim, hidden_dim), gain=1.0),
            nn.ELU(inplace=True),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim), gain=1.0),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs_dict):
        B, C, H, W = obs_dict["screen_image"].shape

        topline = obs_dict["tty_chars"][..., 0, :]
        bottom_line = obs_dict["tty_chars"][..., -2:, :]

        if self.use_learned_embeddings:
            screen_image = obs_dict["screen_image"]
            chars = screen_image[:, 0]
            colors = screen_image[:, 1]
            chars, colors = self._embed(chars, colors)
            screen_image = self._stack(chars, colors)
        else:
            screen_image = obs_dict["screen_image"]

        encodings = [
            self.topline_encoder(topline.float(memory_format=torch.contiguous_format).view(B, -1)),
            self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(B, -1)),
            self.screen_encoder(screen_image.float(memory_format=torch.contiguous_format).view(B, -1, H, W)),
        ]

        if self.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(torch.nn.functional.one_hot(prev_actions, self.num_actions))

        encodings = self.fc(torch.cat(encodings, dim=1))

        return encodings

    def _embed(self, chars, colors):
        chars = selectt(self.char_embeddings, chars.long(), True)
        colors = selectt(self.color_embeddings, colors.long(), True)
        return chars, colors

    def _stack(self, chars, colors):
        obs = torch.cat([chars, colors], dim=-1)
        return obs.permute(0, 3, 1, 2).contiguous()


def selectt(embedding_layer, x, use_index_select):
    """Use index select instead of default forward to possible speed up embedding."""
    if use_index_select:
        out = embedding_layer.weight.index_select(0, x.view(-1))
        # handle reshaping x to 1-d and output back to N-d
        return out.view(x.shape + (-1,))
    else:
        return embedding_layer(x)


class SimBaActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimBaEncoder(
            obs_space=obs_space,
            hidden_dim=self.cfg.actor_hidden_dim,
            depth=self.cfg.actor_depth,
            use_prev_action=self.cfg.use_prev_action,
            use_learned_embeddings=self.cfg.use_learned_embeddings,
            use_max_pool=self.cfg.use_max_pool,
            expansion=self.cfg.expansion,
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
            use_learned_embeddings=self.cfg.use_learned_embeddings,
            use_max_pool=self.cfg.use_max_pool,
            expansion=self.cfg.expansion,
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
    cfg = parse_nethack_args(
        argv=[
            "--env=nethack_score",
            "--add_image_observation=True",
            "--pixel_size=1",
            "--use_learned_embeddings=True",
            "--critic_hidden_dim=512",
            "--critic_depth=3",
            "--use_max_pool=True",
        ]
    )

    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    obs, info = env.reset()
    encoder = SimBaCriticEncoder(cfg, env_info.obs_space)
    print(encoder)
    x = encoder(obs)
    print(x.shape)
