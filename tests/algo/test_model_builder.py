import numpy as np
import pytest
import torch
from gym.spaces import Box, Dict

from sample_factory.cfg.arguments import parse_sf_args
from sample_factory.model.encoder import default_make_encoder_func


class TestModelBuilder:
    @pytest.mark.parametrize(
        "obs_space",
        [
            Dict(
                {
                    "obs_1d": Box(-1, 1, shape=(21,)),
                }
            ),
            Dict(
                {
                    "obs_1d": Box(-1, 1, shape=(21,)),
                    "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
                }
            ),
            Dict(
                {
                    "obs_1d": Box(-1, 1, shape=(21,)),
                    "obs_3d": Box(-1, 1, shape=(3, 84, 84)),
                    "obs_3d_2": Box(-1, 1, shape=(3, 64, 64)),
                }
            ),
            Dict(
                {
                    "obs": Box(-1, 1, shape=(21,)),
                }
            ),
            Dict(
                {
                    "obs": Box(-1, 1, shape=(3, 84, 84)),
                }
            ),
            Dict(
                {
                    "obs": Box(-1, 1, shape=(3, 84, 84)),
                    "measurements": Box(-1, 1, shape=(21,)),
                }
            ),
        ],
    )
    def test_default_make_encoder_func(self, obs_space):
        parser, cfg = parse_sf_args(argv=["--env=dummy"])
        encoder = default_make_encoder_func(cfg, obs_space)
        obs = obs_space.sample()
        for k in obs.keys():
            obs[k] = torch.from_numpy(np.expand_dims(obs[k], 0))

        output = encoder(obs)

        assert set(encoder.obs_keys) == set(obs_space.keys())
        assert output.shape == (1, 512 * len(obs_space))
