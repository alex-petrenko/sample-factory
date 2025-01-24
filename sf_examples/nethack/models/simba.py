import torch
import torch.nn as nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.linear1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()
        self.linear2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x + residual


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_blocks,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim, kernel_size=kernel_size, padding=padding, stride=stride) for _ in range(num_blocks)]
        )
        self.final_norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        x = self.input_layer(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_norm(x)
        return x


class SimbaEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        *,
        char_edim,
        color_edim,
        hidden_dim,
        num_blocks,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        self.char_embeddings = nn.Embedding(256, char_edim)
        self.color_embeddings = nn.Embedding(128, color_edim)
        self.resnet = ResNet(
            char_edim + color_edim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            padding=padding,
        )

        screen_shape = obs_space["tty_chars"].shape
        self.out_size = calc_num_elements(self.resnet, (char_edim + color_edim,) + screen_shape)

        self.fc_head = nn.Sequential(
            nn.Linear(self.out_size, hidden_dim), nn.ReLU(inplace=True), nn.LayerNorm(hidden_dim)
        )

    def forward(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        chars, colors = self._embed(chars, colors)
        x = self._stack(chars, colors)
        x = self.resnet(x)
        x = x.view(-1, self.out_size)
        x = self.fc_head(x)
        return x

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
        # Access weight through the embedding layer
        return nn.functional.embedding(x, embedding_layer.weight)
    else:
        # Use standard embedding forward
        return embedding_layer(x)


class SimbaActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimbaEncoder(
            obs_space=obs_space,
            char_edim=self.cfg.actor_char_edim,
            color_edim=self.cfg.actor_color_edim,
            hidden_dim=self.cfg.actor_hidden_dim,
            num_blocks=self.cfg.actor_num_blocks,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.cfg.actor_hidden_dim


class SimbaCriticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = SimbaEncoder(
            obs_space=obs_space,
            char_edim=self.cfg.critic_char_edim,
            color_edim=self.cfg.critic_color_edim,
            hidden_dim=self.cfg.critic_hidden_dim,
            num_blocks=self.cfg.critic_num_blocks,
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
    cfg = parse_nethack_args(argv=["--env=nethack_score"])

    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    obs, info = env.reset()
    encoder = SimbaCriticEncoder(cfg, env_info.obs_space)
    x = encoder(obs)
    print(x.shape)
