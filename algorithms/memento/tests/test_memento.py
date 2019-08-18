from unittest import TestCase

import torch

from algorithms.memento.mem_wrapper import MemCategorical


class TestMemCategorical(TestCase):
    def test_mem_categorical(self):
        logits = torch.ones([3, 2])
        prior_probs = [0.5, 0.5]
        distr = MemCategorical(logits, prior_probs)

        kl = distr.entropy()
        self.assertEqual(list(kl.shape), [3])
        self.assertEqual(kl.cpu().tolist(), [0, 0, 0])

        logits = torch.ones([3, 2])
        prior_probs = [0.1, 0.9]
        distr = MemCategorical(logits, prior_probs)
        kl_1 = distr.entropy()
        self.assertEqual(list(kl_1.shape), [3])

        logits = torch.ones([3, 2])
        prior_probs = [0.01, 0.99]
        distr = MemCategorical(logits, prior_probs)
        kl_2 = distr.entropy()
        self.assertEqual(list(kl_2.shape), [3])

        self.assertGreater(-kl_2[0].item(), -kl_1[0].item())
