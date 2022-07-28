import imp

import numpy as np
import pytest
import torch
from gym.spaces import Box, Dict

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.model.model_utils import create_standard_encoder


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
    encoder = create_standard_encoder(cfg, obs_space, None)
    obs = obs_space.sample()
    for k in obs.keys():
        obs[k] = torch.from_numpy(np.expand_dims(obs[k], 0))

    output = encoder(obs)

    print(encoder)
    print(output.size())


if __name__ == "__main__":
    test_default_encoder()
