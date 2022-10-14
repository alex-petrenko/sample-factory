import time

import gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from sample_factory.algo.utils.action_distributions import (
    calc_num_action_parameters,
    calc_num_actions,
    get_action_distribution,
    sample_actions_log_probs,
)
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log


class TestActionDistributions:
    @pytest.mark.parametrize("gym_space", [gym.spaces.Discrete(3)])
    @pytest.mark.parametrize("batch_size", [128])
    def test_simple_distribution(self, gym_space, batch_size):
        simple_action_space = gym_space
        simple_num_logits = calc_num_action_parameters(simple_action_space)
        assert simple_num_logits == simple_action_space.n

        simple_logits = torch.rand(batch_size, simple_num_logits)
        simple_action_distribution = get_action_distribution(simple_action_space, simple_logits)

        simple_actions = simple_action_distribution.sample()
        assert list(simple_actions.shape) == [batch_size, 1]
        assert all(0 <= a < simple_action_space.n for a in simple_actions)

    @pytest.mark.parametrize("gym_space", [gym.spaces.Discrete(3)])
    @pytest.mark.parametrize("batch_size", [128])
    @pytest.mark.parametrize("device_type", ["cpu"])
    def test_gumbel_trick(self, gym_space, batch_size, device_type):
        """
        We use a Gumbel noise which seems to be faster compared to using pytorch multinomial.
        Here we test that those are actually equivalent.
        """

        timing = Timing()

        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.enabled = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            action_space = gym_space
            num_logits = calc_num_action_parameters(action_space)
            device = torch.device(device_type)
            logits = torch.rand(batch_size, num_logits, device=device) * 10.0 - 5.0

            if device_type == "cuda":
                torch.cuda.synchronize(device)

            count_gumbel, count_multinomial = np.zeros([action_space.n]), np.zeros([action_space.n])

            # estimate probability mass by actually sampling both ways
            num_samples = 20000

            action_distribution = get_action_distribution(action_space, logits)
            sample_actions_log_probs(action_distribution)
            action_distribution.sample_gumbel()

            with timing.add_time("gumbel"):
                for i in range(num_samples):
                    action_distribution = get_action_distribution(action_space, logits)
                    samples_gumbel = action_distribution.sample_gumbel()
                    count_gumbel[samples_gumbel[0]] += 1

            action_distribution = get_action_distribution(action_space, logits)
            action_distribution.sample()

            with timing.add_time("multinomial"):
                for i in range(num_samples):
                    action_distribution = get_action_distribution(action_space, logits)
                    samples_multinomial = action_distribution.sample()
                    count_multinomial[samples_multinomial[0]] += 1

            estimated_probs_gumbel = count_gumbel / float(num_samples)
            estimated_probs_multinomial = count_multinomial / float(num_samples)

            log.debug("Gumbel estimated probs: %r", estimated_probs_gumbel)
            log.debug("Multinomial estimated probs: %r", estimated_probs_multinomial)
            log.debug("Sampling timing: %s", timing)
            time.sleep(0.1)  # to finish logging

    @pytest.mark.parametrize("num_spaces", [1, 4])
    @pytest.mark.parametrize("gym_space", [gym.spaces.Discrete(1), gym.spaces.Discrete(3)])
    @pytest.mark.parametrize("batch_size", [128])
    def test_tuple_distribution(self, num_spaces, gym_space, batch_size):
        spaces = [gym_space for _ in range(num_spaces)]
        action_space = gym.spaces.Tuple(spaces)

        num_logits = calc_num_action_parameters(action_space)
        logits = torch.rand(batch_size, num_logits)

        assert num_logits == sum(s.n for s in action_space.spaces)

        action_distribution = get_action_distribution(action_space, logits)

        tuple_actions = action_distribution.sample()
        assert list(tuple_actions.shape) == [batch_size, num_spaces]

        log_probs = action_distribution.log_prob(tuple_actions)
        assert list(log_probs.shape) == [batch_size]

        entropy = action_distribution.entropy()
        assert list(entropy.shape) == [batch_size]

    @pytest.mark.parametrize("num_spaces", [3])
    @pytest.mark.parametrize("num_actions", [2])
    def test_tuple_sanity_check(self, num_spaces, num_actions):
        simple_space = gym.spaces.Discrete(num_actions)
        spaces = [simple_space for _ in range(num_spaces)]
        tuple_space = gym.spaces.Tuple(spaces)

        assert calc_num_action_parameters(tuple_space) == num_spaces * num_actions

        simple_logits = torch.zeros(1, num_actions)
        tuple_logits = torch.zeros(1, calc_num_action_parameters(tuple_space))

        simple_distr = get_action_distribution(simple_space, simple_logits)
        tuple_distr = get_action_distribution(tuple_space, tuple_logits)

        tuple_entropy = tuple_distr.entropy()
        assert tuple_entropy == simple_distr.entropy() * num_spaces

        simple_logprob = simple_distr.log_prob(torch.ones(1, 1))
        tuple_logprob = tuple_distr.log_prob(torch.ones(1, num_spaces))
        assert tuple_logprob == simple_logprob * num_spaces

    def test_sanity(self):
        raw_logits = torch.tensor([[0.0, 1.0, 2.0]])
        action_space = gym.spaces.Discrete(3)
        categorical = get_action_distribution(action_space, raw_logits)

        torch_categorical = Categorical(logits=raw_logits)
        torch_categorical_log_probs = torch_categorical.log_prob(torch.tensor([0, 1, 2]))

        entropy = categorical.entropy()
        torch_entropy = torch_categorical.entropy()
        assert np.allclose(entropy.numpy(), torch_entropy)

        log_probs = categorical.log_prob(torch.tensor([[0, 1, 2]]))

        assert np.allclose(torch_categorical_log_probs.numpy(), log_probs.numpy())

        probs = torch.exp(log_probs)

        expected_probs = np.array([0.09003057317038046, 0.24472847105479764, 0.6652409557748219])

        assert np.allclose(probs.numpy(), expected_probs)

        tuple_space = gym.spaces.Tuple([action_space, action_space])
        raw_logits = torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]])
        tuple_distr = get_action_distribution(tuple_space, raw_logits)

        for a1 in [0, 1, 2]:
            for a2 in [0, 1, 2]:
                action = torch.tensor([[a1, a2]])
                log_prob = tuple_distr.log_prob(action)
                probability = torch.exp(log_prob)[0].item()
                assert probability == pytest.approx(expected_probs[a1] * expected_probs[a2], 1e-6)


@pytest.mark.parametrize(
    "spaces",
    [
        [gym.spaces.Discrete, gym.spaces.Discrete],
        [gym.spaces.Discrete, gym.spaces.Box],
        [gym.spaces.Box, gym.spaces.Box],
    ],
)
@pytest.mark.parametrize("sizes", [[1, 1], [2, 1], [1, 2], [2, 3]])
def test_tuple_action_distribution(spaces, sizes):
    # I like to use prime numbers for tests as it can flag problems hidden by automatic broadcasting etc
    BATCH_SIZE = 31

    assert len(spaces) > 0
    assert len(spaces) == len(sizes)

    num_actions = 0
    num_logits = 0

    _action_spaces = []
    for space, size in zip(spaces, sizes):
        if space is gym.spaces.Discrete:
            _action_spaces.append(space(size))
            num_actions += 1
            num_logits += size
        else:
            _action_spaces.append(gym.spaces.Box(low=-1, high=1, shape=(size,), dtype=np.float32))
            num_actions += size
            num_logits += size * 2

    action_space = gym.spaces.Tuple(_action_spaces)

    assert calc_num_actions(action_space) == num_actions
    assert calc_num_action_parameters(action_space) == num_logits

    logits = torch.randn(BATCH_SIZE, num_logits)
    action_dist = get_action_distribution(action_space, logits)

    actions = action_dist.sample()
    assert actions.size() == (BATCH_SIZE, num_actions)

    action_log_probs = action_dist.log_prob(actions)
    assert action_log_probs.size() == (BATCH_SIZE,)

    entropy = action_dist.entropy()
    assert entropy.size() == (BATCH_SIZE,)

    actions, action_log_probs = action_dist.sample_actions_log_probs()

    assert actions.size() == (BATCH_SIZE, num_actions)
    assert action_log_probs.size() == (BATCH_SIZE,)
