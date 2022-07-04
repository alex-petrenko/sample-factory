from typing import Optional, Tuple

import pytest
import torch
from torch import Tensor

from sample_factory.algo.utils.misc import EPS
from sample_factory.algo.utils.running_mean_std import RunningMeanStdInPlace


class TestRMS:
    @pytest.mark.parametrize("batch_size", [2, 100])
    @pytest.mark.parametrize("shape", [(1,), (3, 7)])
    @pytest.mark.parametrize("norm_only", [False, True])
    def test_rms_sanity(
        self,
        batch_size: int,
        shape: Tuple,
        norm_only: bool,
        data: Optional[Tensor] = None,
        use_jit: bool = False,
    ):
        normalizer = RunningMeanStdInPlace(shape, norm_only=norm_only)
        assert normalizer.input_shape == shape

        if use_jit:
            normalizer = torch.jit.script(normalizer)

        # generate some non-normally distributed data
        if data is None:
            data = torch.rand((batch_size,) + shape)
        else:
            assert data.shape == (batch_size,) + shape

        orig_data = data.clone()

        # calculate data statistics
        orig_mean = data.mean()
        std = data.std(dim=0)

        # normalize data
        normalizer.train()  # switch the normalizer to training mode to update the running mean and std
        normalizer(data)  # in-place normalization

        # check that the data is more normalized now
        if not norm_only:
            assert torch.abs(data.mean()) < torch.abs(orig_mean)

        assert torch.all(torch.abs(data.std(dim=0) - 1) < torch.abs(std - 1)).item()

        data_denorm = data.clone()
        normalizer.eval()
        normalizer(data_denorm, denormalize=True)

        if torch.abs(data).max() < normalizer.clip - EPS:
            # this might not be true if we clipped some data
            assert torch.allclose(orig_data, data_denorm, atol=1e-6)

    def test_clip(self):
        self.test_rms_sanity(
            batch_size=1000,
            shape=(1,),
            norm_only=False,
            data=torch.tensor([0.0] * 999 + [1e4]).unsqueeze_(1),
        )

    def test_jit(self):
        self.test_rms_sanity(batch_size=10, shape=(1,), norm_only=False, use_jit=True)
