import numpy as np
import pytest
import torch
import torch.nn as nn

from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs


class TestPackedSequences:
    @staticmethod
    def check_packed_version_matching_loopy_version(T, N, D, random_dones, norm_tolerance=4e-6):
        rnn = nn.GRU(D, D, 1)

        for _ in range(100):
            if random_dones:
                dones = torch.randint(0, 2, (N * T,))
            else:
                dones = torch.zeros((N * T,))
                for i in range(1, N * T, 7):
                    dones[i] = 1.0

            rnn_hidden_states_random = torch.rand(T * N, D)

            x = torch.randn(T * N, D)
            rnn_hidden_states = rnn_hidden_states_random.clone().detach()
            x_seq, seq_states, inverted_select_inds = build_rnn_inputs(
                x,
                dones,
                rnn_hidden_states,
                T,
            )

            packed_out, _ = rnn(x_seq, seq_states.unsqueeze(0))
            packed_out = build_core_out_from_seq(packed_out, inverted_select_inds)

            rnn_hidden_states = rnn_hidden_states_random.clone().detach()
            rnn_hidden_states = rnn_hidden_states[::T].unsqueeze(0)
            loopy_out = []
            for t in range(T):
                rnn_out, rnn_hidden_states = rnn(x[t::T].view(1, N, -1), rnn_hidden_states)
                loopy_out.append(rnn_out.view(N, -1))
                rnn_hidden_states = rnn_hidden_states * (1 - dones[t::T].view(1, N, 1))

            loopy_out = torch.stack(loopy_out, dim=1).view(N * T, -1)

            norm = torch.norm(packed_out - loopy_out)

            assert norm < norm_tolerance
            assert np.allclose(packed_out.detach().numpy(), loopy_out.detach().numpy(), atol=4e-6)

    @pytest.mark.parametrize("T", [37])
    @pytest.mark.parametrize("N", [64])
    @pytest.mark.parametrize("D", [42])
    @pytest.mark.parametrize("random_dones", [True, False])
    @pytest.mark.parametrize("norm_tolerance", [9e-6])
    def test_full_with_larger_param(self, T, N, D, random_dones, norm_tolerance):
        self.check_packed_version_matching_loopy_version(T, N, D, random_dones, norm_tolerance)

    @pytest.mark.parametrize("T", [5, 27])
    @pytest.mark.parametrize("N", [1, 64])
    @pytest.mark.parametrize("D", [1, 10])
    @pytest.mark.parametrize("random_dones", [True, False])
    def test_trivial(self, T, N, D, random_dones):
        self.check_packed_version_matching_loopy_version(T, N, D, random_dones)
