import numpy as np
import torch
from gym.spaces import Box, Dict

from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.model.encoder import default_make_encoder_func


def test_default_encoder():
    parser, cfg = parse_sf_args(argv=["--env=dummy"])
    cfg.encoder_type = "default"
    obs_space = Dict(
        {
            "obs_1d": Box(-1, 1, shape=(21,)),
            "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
            "obs_3d_2": Box(-1, 1, shape=(3, 64, 64)),
        }
    )
    encoder = default_make_encoder_func(cfg, obs_space)
    obs = obs_space.sample()
    for k in obs.keys():
        obs[k] = torch.from_numpy(np.expand_dims(obs[k], 0))

    output = encoder(obs)

    print(encoder)
    print(output.size())


if __name__ == "__main__":
    test_default_encoder()
