from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.tensor_utils import ensure_torch_tensor
from sample_factory.utils.typing import Config


def trajectories_per_minibatch(cfg: Config) -> int:
    return cfg.batch_size // cfg.rollout


def trajectories_per_training_iteration(cfg: Config) -> int:
    return cfg.num_batches_per_epoch * trajectories_per_minibatch(cfg)


def total_num_envs(cfg: Config) -> int:
    return cfg.num_workers * cfg.num_envs_per_worker


def total_num_agents(cfg: Config, env_info: EnvInfo) -> int:
    return total_num_envs(cfg) * env_info.num_agents


def num_agents_per_worker(cfg: Config, env_info: EnvInfo) -> int:
    return cfg.num_envs_per_worker * env_info.num_agents


def prepare_and_normalize_obs(model: Module, obs: TensorDict | Dict[str, Tensor]) -> TensorDict | Dict[str, Tensor]:
    for key, x in obs.items():
        obs[key] = ensure_torch_tensor(x).to(model.device_for_input_tensor(key))
    normalized_obs = model.normalize_obs(obs)
    for key, x in normalized_obs.items():
        normalized_obs[key] = x.type(model.type_for_input_tensor(key))
    return normalized_obs


# noinspection NonAsciiCharacters
def gae_advantages(rewards: Tensor, dones: Tensor, values: Tensor, γ: float, λ: float) -> Tensor:
    rewards = rewards.transpose(0, 1)  # [E, T] -> [T, E]
    dones = dones.transpose(0, 1).float()  # [E, T] -> [T, E]
    values = values.transpose(0, 1)  # [E, T+1] -> [T+1, E]

    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    # section 3 in GAE paper: calculating advantages
    deltas = rewards + (1 - dones) * (γ * values[1:]) - values[:-1]
    advantages = calculate_discounted_sum_torch(deltas, dones, γ * λ)

    # transpose advantages back to [E, T] before creating a single experience buffer
    advantages.transpose_(0, 1)
    return advantages


def calculate_discounted_sum_torch(
    x: Tensor, dones: Tensor, discount: float, x_last: Optional[Tensor] = None
) -> Tensor:
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
