from typing import Tuple, Optional

import torch
from torch import Tensor


def gae_advantages_returns(
        rewards: Tensor, dones: Tensor, values: Tensor, next_values: Tensor,
        gamma: float, gae_lambda: float
) -> Tuple[Tensor, Tensor]:
    # append values estimated based on the last observation
    values = torch.cat((values, next_values.unsqueeze(-1)), dim=-1)  # [E, T] -> [E, T+1]

    rewards = rewards.transpose(0, 1)  # [E, T] -> [T, E]
    dones = dones.transpose(0, 1).float()  # [E, T] -> [T, E]
    values = values.transpose(0, 1)  # [E, T+1] -> [T+1, E]

    advantages, returns = calculate_gae_torch(rewards, dones, values, gamma, gae_lambda)

    # transpose tensors back to [E, T] before creating a single experience buffer
    advantages.transpose_(0, 1)
    returns.transpose_(0, 1)
    return advantages, returns


def calculate_gae_torch(rewards: Tensor, dones: Tensor, values: Tensor, gamma: float, gae_lambda: float) -> Tuple[Tensor, Tensor]:
    """
    Computing discounted cumulative sum, taking episode terminations into consideration. Follows the
    Generalized Advantage Estimation algorithm.

    :param rewards: actual environment rewards
    :param dones: True if absorbing state is reached
    :param values: estimated values
    :param gamma: discount factor [0,1]
    :param gae_lambda: lambda-factor for GAE (discounting for longer-horizon advantage estimations), [0,1]
    :return: advantages and discounted returns
    """
    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    # section 3 in GAE paper: calculating advantages
    deltas = rewards + (1 - dones) * (gamma * values[1:]) - values[:-1]
    advantages = calculate_discounted_sum_torch(deltas, dones, gamma * gae_lambda)

    # targets for value function - this is just a simple discounted sum of rewards
    discounted_returns = calculate_discounted_sum_torch(rewards, dones, gamma, values[-1])

    return advantages, discounted_returns


def calculate_discounted_sum_torch(x: Tensor, dones: Tensor, discount: float, x_last: Optional[Tensor] = None):
    """
    Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
    :param x: ndarray of shape [num_steps, num_envs]
    :param dones: ndarray of shape [num_steps, num_envs]
    :param discount: float in range [0,1]
    :param x_last: iterable of shape [num_envs], value at the end of trajectory. None interpreted as zero(s).
    """
    if x_last is None:
        x_last = x[-1].clone().fill_(0.0)

    cumulative = x_last

    discounted_sum = torch.zeros_like(x)
    for i in reversed(range(len(x))):
        cumulative = x[i] + discount * cumulative * (1.0 - dones[i])
        discounted_sum[i] = cumulative
    return discounted_sum
