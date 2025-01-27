import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.utils import he_normal_init, orthogonal_init
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.scaled import BottomLinesEncoder, TopLineEncoder

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head), FeedForward(dim, mlp_dim)])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
    ):
        super().__init__()

        self.dim = dim

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        # we want the embeddings, not classification
        # self.linear_head = nn.Linear(dim, num_classes)

    def get_out_size(self):
        return self.dim

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(x.device, dtype=x.dtype)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.to_latent(x)
        # return self.linear_head(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        obs_space,
        hidden_dim,
        depth,
        heads,
        mlp_dim,
        use_prev_action: bool = True,
        use_learned_embeddings: bool = False,
        char_edim: int = 16,
        color_edim: int = 16,
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

        C, W, H = obs_space["screen_image"].shape
        self.screen_encoder = SimpleViT(
            image_size=(W, H),
            patch_size=(3, 3),
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=C,
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


class ViTActorEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = ViTEncoder(
            obs_space=obs_space,
            hidden_dim=self.cfg.actor_hidden_dim,
            depth=self.cfg.actor_depth,
            heads=self.cfg.actor_heads,
            mlp_dim=self.cfg.actor_mlp_dim,
            use_prev_action=self.cfg.use_prev_action,
        )

    def forward(self, x):
        return self.model(x)

    def get_out_size(self):
        return self.cfg.actor_hidden_dim


class ViTCriticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.model = ViTEncoder(
            obs_space=obs_space,
            hidden_dim=self.cfg.critic_hidden_dim,
            depth=self.cfg.critic_depth,
            heads=self.cfg.critic_heads,
            mlp_dim=self.cfg.critic_mlp_dim,
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
    cfg = parse_nethack_args(
        argv=[
            "--env=nethack_score",
            "--add_image_observation=True",
            "--pixel_size=1",
            "--use_learned_embeddings=True",
            "--critic_hidden_dim=512",
            "--critic_depth=3",
        ]
    )
    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    env_info = extract_env_info(env, cfg)

    obs, info = env.reset()
    encoder = ViTCriticEncoder(cfg, env_info.obs_space)
    print(encoder)
    x = encoder(obs)
    print(x.shape)
