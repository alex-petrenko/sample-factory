import random
from unittest import TestCase

import gym
import numpy as np
import torch

from algorithms.utils.action_distributions import get_action_distribution, calc_num_logits


class TestActionDistributions(TestCase):
    batch_size = 42  # whatever

    def test_simple_distribution(self):
        simple_action_space = gym.spaces.Discrete(3)
        simple_num_logits = calc_num_logits(simple_action_space)
        self.assertEqual(simple_num_logits, simple_action_space.n)

        simple_logits = torch.rand(self.batch_size, simple_num_logits)
        simple_action_distribution = get_action_distribution(simple_action_space, simple_logits)

        simple_actions = simple_action_distribution.sample()
        self.assertEqual(list(simple_actions.shape), [self.batch_size])
        self.assertTrue(all(0 <= a < simple_action_space.n for a in simple_actions))

    def test_tuple_distribution(self):
        num_spaces = random.randint(1, 4)
        spaces = [gym.spaces.Discrete(random.randint(2, 5)) for _ in range(num_spaces)]
        action_space = gym.spaces.Tuple(spaces)

        num_logits = calc_num_logits(action_space)
        logits = torch.rand(self.batch_size, num_logits)

        self.assertEqual(num_logits, sum(s.n for s in action_space.spaces))

        action_distribution = get_action_distribution(action_space, logits)

        tuple_actions = action_distribution.sample()
        self.assertEqual(list(tuple_actions.shape), [self.batch_size, num_spaces])

        log_probs = action_distribution.log_prob(tuple_actions)
        self.assertEqual(list(log_probs.shape), [self.batch_size])

        entropy = action_distribution.entropy()
        self.assertEqual(list(entropy.shape), [self.batch_size])

    def test_tuple_sanity_check(self):
        num_spaces, num_actions = 3, 2
        simple_space = gym.spaces.Discrete(num_actions)
        spaces = [simple_space for _ in range(num_spaces)]
        tuple_space = gym.spaces.Tuple(spaces)

        self.assertTrue(calc_num_logits(tuple_space), num_spaces * num_actions)

        simple_logits = torch.zeros(1, num_actions)
        tuple_logits = torch.zeros(1, calc_num_logits(tuple_space))

        simple_distr = get_action_distribution(simple_space, simple_logits)
        tuple_distr = get_action_distribution(tuple_space, tuple_logits)

        tuple_entropy = tuple_distr.entropy()
        self.assertEqual(tuple_entropy, simple_distr.entropy() * num_spaces)

        simple_logprob = simple_distr.log_prob(torch.ones(1))
        tuple_logprob = tuple_distr.log_prob(torch.ones(1, num_spaces))
        self.assertEqual(tuple_logprob, simple_logprob * num_spaces)

    def test_sanity(self):
        raw_logits = torch.tensor([[0.0, 1.0, 2.0]])
        action_space = gym.spaces.Discrete(3)
        categorical = get_action_distribution(action_space, raw_logits)

        log_probs = categorical.log_prob(torch.tensor([0, 1, 2]))
        probs = torch.exp(log_probs)

        expected_probs = np.array([0.09003057317038046, 0.24472847105479764, 0.6652409557748219])

        self.assertTrue(np.allclose(probs.numpy(), expected_probs))

        tuple_space = gym.spaces.Tuple([action_space, action_space])
        raw_logits = torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]])
        tuple_distr = get_action_distribution(tuple_space, raw_logits)

        for a1 in [0, 1, 2]:
            for a2 in [0, 1, 2]:
                action = torch.tensor([[a1, a2]])
                log_prob = tuple_distr.log_prob(action)
                probability = torch.exp(log_prob)[0].item()
                self.assertAlmostEqual(probability, expected_probs[a1] * expected_probs[a2], delta=1e-6)


