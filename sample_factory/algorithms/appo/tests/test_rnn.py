from unittest import TestCase

import numpy as np

import torch
import torch.nn as nn

from sample_factory.algorithms.appo.learner import build_rnn_inputs, build_core_out_from_seq


# noinspection PyPep8Naming
class TestPackedSequences(TestCase):
    def check_packed_version_matching_loopy_version(self, T, N, D, random_dones):
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
                x, dones, rnn_hidden_states, T,
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
            print(norm)
            self.assertLess(norm, 2e-6)
            self.assertTrue(np.allclose(packed_out.detach().numpy(), loopy_out.detach().numpy(), atol=2e-6))

    def test_full(self):
        T = 37  # recurrence, bptt
        N = 64  # batch size
        D = 42  # RNN cell size (hidden state size)
        self.check_packed_version_matching_loopy_version(T, N, D, True)
        self.check_packed_version_matching_loopy_version(T, N, D, False)

    def test_trivial(self):
        T = 5  # recurrence, bptt
        N = 1  # batch size
        D = 1  # RNN cell size (hidden state size)
        self.check_packed_version_matching_loopy_version(T, N, D, True)
        self.check_packed_version_matching_loopy_version(T, N, D, False)

