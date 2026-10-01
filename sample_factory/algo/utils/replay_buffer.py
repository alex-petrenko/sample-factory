from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.typing import Device
from sample_factory.utils.utils import log


# https://github.com/Kaixhin/Rainbow/
class SumSegmentTree:
    def __init__(self, capacity: int):
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity *= 2

        # Tree has 2*capacity - 1 nodes, but we use 2*capacity for 1-indexed
        self._tree = np.zeros(2 * self._capacity, dtype=np.float64)

    def update(self, idx: int, value: float) -> None:
        idx += self._capacity
        self._tree[idx] = value

        while idx > 1:
            idx //= 2
            self._tree[idx] = self._tree[2 * idx] + self._tree[2 * idx + 1]

    def update_batch(self, indices: np.ndarray, values: np.ndarray) -> None:
        leaf_indices = indices + self._capacity
        self._tree[leaf_indices] = values

        nodes_to_update = set()
        for idx in leaf_indices:
            parent = idx // 2
            while parent >= 1:
                nodes_to_update.add(parent)
                parent //= 2

        # bottom to top
        for node in sorted(nodes_to_update, reverse=True):
            self._tree[node] = self._tree[2 * node] + self._tree[2 * node + 1]

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        if end is None:
            end = self._capacity
        return self._query(start, end, 1, 0, self._capacity)

    def _query(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        if start >= node_end or end <= node_start:
            return 0.0
        if start <= node_start and end >= node_end:
            return self._tree[node]

        mid = (node_start + node_end) // 2
        return self._query(start, end, 2 * node, node_start, mid) + self._query(start, end, 2 * node + 1, mid, node_end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find highest idx: sum(0, idx) <= prefixsum"""
        idx = 1
        while idx < self._capacity:
            left = 2 * idx
            if self._tree[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._tree[left]
                idx = left + 1
        return idx - self._capacity

    def find_prefixsum_idx_batch(self, prefixsums: np.ndarray) -> np.ndarray:
        batch_size = len(prefixsums)
        indices = np.ones(batch_size, dtype=np.int64)
        remaining = prefixsums.copy()

        while np.any(indices < self._capacity):
            mask = indices < self._capacity
            left = 2 * indices
            left_vals = self._tree[left]

            go_left = left_vals > remaining
            indices = np.where(mask & go_left, left, indices)
            remaining = np.where(mask & ~go_left, remaining - left_vals, remaining)
            indices = np.where(mask & ~go_left, left + 1, indices)

        return indices - self._capacity

    def __getitem__(self, idx) -> float:
        if isinstance(idx, np.ndarray):
            return self._tree[idx + self._capacity]
        return self._tree[idx + self._capacity]


class MinSegmentTree:
    def __init__(self, capacity: int):
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity *= 2
        self._tree = np.full(2 * self._capacity, float("inf"), dtype=np.float64)

    def update(self, idx: int, value: float) -> None:
        idx += self._capacity
        self._tree[idx] = value
        while idx > 1:
            idx //= 2
            self._tree[idx] = min(self._tree[2 * idx], self._tree[2 * idx + 1])

    def update_batch(self, indices: np.ndarray, values: np.ndarray) -> None:
        leaf_indices = indices + self._capacity
        self._tree[leaf_indices] = values

        nodes_to_update = set()
        for idx in leaf_indices:
            parent = idx // 2
            while parent >= 1:
                nodes_to_update.add(parent)
                parent //= 2

        for node in sorted(nodes_to_update, reverse=True):
            self._tree[node] = min(self._tree[2 * node], self._tree[2 * node + 1])

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        if end is None:
            end = self._capacity
        return self._query(start, end, 1, 0, self._capacity)

    def _query(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        if start >= node_end or end <= node_start:
            return float("inf")
        if start <= node_start and end >= node_end:
            return self._tree[node]

        mid = (node_start + node_end) // 2
        return min(
            self._query(start, end, 2 * node, node_start, mid),
            self._query(start, end, 2 * node + 1, mid, node_end),
        )

    def __getitem__(self, idx) -> float:
        if isinstance(idx, np.ndarray):
            return self._tree[idx + self._capacity]
        return self._tree[idx + self._capacity]


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_space,
        action_space,
        device: Device = "cpu",  # Store on RAM
        share_memory: bool = True,  # For async
    ):
        self.capacity = capacity
        self.obs_space = obs_space
        self.action_space = action_space
        self.device = torch.device(device)
        self.share_memory = share_memory

        self._ptr = 0
        self._size = 0

        self._storage: Optional[TensorDict] = None
        self._initialized = False

    def _init_storage(self, sample_batch: TensorDict) -> None:
        if self._initialized:
            return

        self._storage = TensorDict()

        def _create_buffer(tensor: Tensor) -> Tensor:
            shape = (self.capacity,) + tensor.shape[1:]  # Replace batch dim with capacity
            dtype = tensor.dtype
            buf = torch.zeros(shape, dtype=dtype, device=self.device)
            if self.share_memory and not buf.is_cuda:
                buf.share_memory_()
            return buf

        # Create buffers for each sample in the sample batch
        for key, value in sample_batch.items():
            if isinstance(value, TensorDict):
                self._storage[key] = TensorDict()
                for k, v in value.items():
                    self._storage[key][k] = _create_buffer(v)
            elif isinstance(value, Tensor):
                self._storage[key] = _create_buffer(value)

        self._initialized = True
        log.debug(f"ReplayBuffer created: {self.capacity}")

    def add(self, batch: TensorDict) -> int:
        """Add batch to buffer"""
        if not self._initialized:
            self._init_storage(batch)

        batch_size = self._get_batch_size(batch)
        if batch_size == 0:
            return 0

        # Copy data to buffer
        if self._ptr + batch_size <= self.capacity:
            self._copy_to_storage(batch, self._ptr, self._ptr + batch_size)
        else:
            first_part = self.capacity - self._ptr
            second_part = batch_size - first_part
            first_batch = self._slice_batch(batch, 0, first_part)
            second_batch = self._slice_batch(batch, first_part, batch_size)

            self._copy_to_storage(first_batch, self._ptr, self.capacity)
            self._copy_to_storage(second_batch, 0, second_part)

        self._ptr = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

        return batch_size

    def _get_batch_size(self, batch: TensorDict) -> int:
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                for k, v in value.items():
                    return v.shape[0]
            elif isinstance(value, Tensor):
                return value.shape[0]
        return 0

    def _slice_batch(self, batch: TensorDict, start: int, end: int) -> TensorDict:
        """Slice along first dim"""
        result = TensorDict()
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                result[key] = TensorDict()
                for k, v in value.items():
                    result[key][k] = v[start:end]
            elif isinstance(value, Tensor):
                result[key] = value[start:end]
        return result

    def _copy_to_storage(self, batch: TensorDict, start: int, end: int) -> None:
        if self._storage is None:
            return
        for key, value in batch.items():
            if isinstance(value, TensorDict):
                for k, v in value.items():
                    self._storage[key][k][start:end].copy_(v.to(self.device))
            elif isinstance(value, Tensor):
                self._storage[key][start:end].copy_(value.to(self.device))

    def sample(self, batch_size: int, device: Optional[Device] = None) -> Optional[TensorDict]:
        if self._size < batch_size or self._storage is None:
            return None

        target_device = device if device is not None else str(self.device)
        indices = torch.randint(0, self._size, (batch_size,), device="cpu")
        return self._index_storage(indices, target_device)

    def _index_storage(self, indices: Tensor, device: Device) -> TensorDict:
        result = TensorDict()
        target_device = torch.device(device)

        if self._storage is None:
            return result

        # Index w/ indices and move to device
        for key, value in self._storage.items():
            if isinstance(value, TensorDict):
                result[key] = TensorDict()
                for k, v in value.items():
                    result[key][k] = v[indices].to(target_device)
            elif isinstance(value, Tensor):
                result[key] = value[indices].to(target_device)

        return result

    def __len__(self) -> int:
        return self._size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    https://arxiv.org/abs/1511.05952 https://projector-video-pdf-converter.datacamp.com/36398/chapter2.pdf
    """

    def __init__(
        self,
        capacity: int,
        obs_space,
        action_space,
        omega: float = 0.6,
        beta_start: float = 0.4,
        device: Device = "cpu",
        share_memory: bool = True,
        epsilon: float = 1e-6,
    ):
        super().__init__(capacity, obs_space, action_space, device, share_memory)

        self.omega = omega
        self.beta = beta_start
        self.epsilon = epsilon
        self._sum_tree = SumSegmentTree(capacity)
        self._min_tree = MinSegmentTree(capacity)
        self._max_priority = 1.0

    def add(self, batch: TensorDict) -> int:
        if not self._initialized:
            self._init_storage(batch)

        batch_size = self._get_batch_size(batch)
        if batch_size == 0:
            return 0

        start_idx = self._ptr

        if self._ptr + batch_size <= self.capacity:
            self._copy_to_storage(batch, self._ptr, self._ptr + batch_size)
            indices = np.arange(start_idx, start_idx + batch_size)
        else:
            first_part = self.capacity - self._ptr
            second_part = batch_size - first_part
            first_batch = self._slice_batch(batch, 0, first_part)
            second_batch = self._slice_batch(batch, first_part, batch_size)

            self._copy_to_storage(first_batch, self._ptr, self.capacity)
            self._copy_to_storage(second_batch, 0, second_part)
            indices = np.concatenate([np.arange(start_idx, self.capacity), np.arange(0, second_part)])

        # Batch update priorities
        priority = self._max_priority**self.omega
        priorities = np.full(len(indices), priority, dtype=np.float64)
        self._sum_tree.update_batch(indices, priorities)
        self._min_tree.update_batch(indices, priorities)

        self._ptr = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)
        return batch_size

    def sample(self, batch_size: int, device: Optional[Device] = None):
        if self._size < batch_size or self._storage is None:
            return None

        target_device = device if device is not None else str(self.device)
        indices = self._sample_proportional(batch_size)
        weights = self._compute_is_weights(indices, target_device)
        indices_tensor = torch.from_numpy(indices).long()
        batch = self._index_storage(indices_tensor, target_device)
        return batch, weights, indices_tensor.to(target_device)

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        total_priority = self._sum_tree.sum(0, self._size)
        segment = total_priority / batch_size

        # Generate all random samples at once
        segment_starts = np.arange(batch_size) * segment
        segment_ends = segment_starts + segment
        prefixsums = np.random.uniform(segment_starts, segment_ends)

        # Batch find indices
        indices = self._sum_tree.find_prefixsum_idx_batch(prefixsums)
        indices = np.clip(indices, 0, self._size - 1)
        return indices

    def _compute_is_weights(self, indices: np.ndarray, device: Device) -> Tensor:
        """
        Importance w_i = (N * P(i))^(-beta) / max_j(w_j) = (N * (priority_i / sum(priorities)))^(-beta) / max_j(w_j)
        """
        total_priority = self._sum_tree.sum(0, self._size)
        min_priority = self._min_tree.min(0, self._size)

        max_weight = (self._size * min_priority / total_priority) ** (-self.beta)

        priorities = self._sum_tree[indices]
        probs = priorities / total_priority
        weights = (self._size * probs) ** (-self.beta)
        weights = weights / max_weight

        return torch.from_numpy(weights.astype(np.float32)).to(device)

    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None:
        indices_np = indices.cpu().numpy().astype(np.int64)
        priorities_np = priorities.cpu().numpy()

        # Vectorized priority computation
        priorities_np = np.maximum(priorities_np + self.epsilon, self.epsilon)
        self._max_priority = max(self._max_priority, float(priorities_np.max()))

        priority_omega = priorities_np**self.omega
        self._sum_tree.update_batch(indices_np, priority_omega)
        self._min_tree.update_batch(indices_np, priority_omega)

    def set_beta(self, beta: float) -> None:
        self.beta = beta
