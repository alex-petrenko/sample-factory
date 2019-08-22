from unittest import TestCase

import torch

from algorithms.utils.action_distributions import CategoricalActionDistribution


class TestMemCategorical(TestCase):
    def test_mem_categorical(self):
        logits = torch.ones([3, 2])
        prior_probs = [0.5, 0.5]
        distr = CategoricalActionDistribution(logits, prior_probs)

        kl = distr.kl_prior()
        self.assertEqual(list(kl.shape), [3])
        self.assertEqual(kl.cpu().tolist(), [0, 0, 0])

        logits = torch.ones([3, 2])
        prior_probs = [0.1, 0.9]
        distr = CategoricalActionDistribution(logits, prior_probs)
        kl_1 = distr.kl_prior()
        self.assertEqual(list(kl_1.shape), [3])

        logits = torch.ones([3, 2])
        prior_probs = [0.01, 0.99]
        distr = CategoricalActionDistribution(logits, prior_probs)
        kl_2 = distr.kl_prior()
        self.assertEqual(list(kl_2.shape), [3])

        self.assertGreater(kl_2[0].item(), kl_1[0].item())

        kl_3 = distr.kl_divergence(distr)
        self.assertEqual(kl_3.cpu().tolist(), [0, 0, 0])
