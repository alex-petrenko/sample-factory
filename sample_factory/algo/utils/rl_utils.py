from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
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


def samples_per_trajectory(trajectory: TensorDict) -> int:
    shape = trajectory["rewards"].shape
    batch, rollout = shape[0], shape[1]
    return batch * rollout


@torch.jit.script
def calculate_discounted_sum_torch(
    x: Tensor, dones: Tensor, valids: Tensor, discount: float, x_last: Optional[Tensor] = None
) -> Tensor:
    """
    Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
    """
    if x_last is None:
        x_last = x[-1].clone().fill_(0.0)

    cumulative = x_last

    discounted_sum = torch.zeros_like(x)
    i = len(x) - 1
    while i >= 0:
        # do not discount invalid steps so we can entirely skip a part of the trajectory
        # x should be already multiplied by valids
        discount_valid = discount * valids[i] + (1 - valids[i])
        cumulative = x[i] + discount_valid * cumulative * (1.0 - dones[i])
        discounted_sum[i] = cumulative
        i -= 1

    return discounted_sum


# noinspection NonAsciiCharacters
@torch.jit.script
def gae_advantages(rewards: Tensor, dones: Tensor, values: Tensor, valids: Tensor, γ: float, λ: float) -> Tensor:
    rewards = rewards.transpose(0, 1)  # [E, T] -> [T, E]
    dones = dones.transpose(0, 1).float()  # [E, T] -> [T, E]
    values = values.transpose(0, 1)  # [E, T+1] -> [T+1, E]
    valids = valids.transpose(0, 1).float()  # [E, T+1] -> [T+1, E]

    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    # section 3 in GAE paper: calculating advantages
    deltas = (rewards - values[:-1]) * valids[:-1] + (1 - dones) * (γ * values[1:] * valids[1:])

    advantages = calculate_discounted_sum_torch(deltas, dones, valids[:-1], γ * λ)

    # transpose advantages back to [E, T] before creating a single experience buffer
    advantages.transpose_(0, 1)
    return advantages


DonesType = Union[bool, np.ndarray, Tensor, Sequence[bool]]


def make_dones(terminated: DonesType, truncated: DonesType) -> DonesType:
    """
    Make dones from terminated/truncated (gym 0.26.0 changes).
    Assumes that terminated and truncated are the same type and shape.
    """
    if isinstance(terminated, (bool, np.ndarray, Tensor)):
        return terminated | truncated
    elif isinstance(terminated, Sequence):
        return [t | truncated[i] for i, t in enumerate(terminated)]

    raise ValueError(f"Unknown {type(terminated)=}")
