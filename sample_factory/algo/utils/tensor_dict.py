from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.misc import MAGIC_FLOAT, MAGIC_INT
from sample_factory.utils.dicts import (
    copy_dict_structure,
    iter_dicts_recursively,
    iterate_recursively,
    list_of_dicts_to_dict_of_lists,
)
from sample_factory.utils.utils import log


class TensorDict(dict):
    dict_key_type = str

    def __getitem__(self, key):
        if isinstance(key, self.dict_key_type):
            # if key is string assume we're accessing dict's interface
            return dict.__getitem__(self, key)
        else:
            # otherwise we want to index/slice into tensors themselves
            return self._index_func(self, key)

    def _index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self._index_func(value, indices)
            return res
        else:
            t = x[indices]
            return t

    def __setitem__(self, key, value):
        if isinstance(key, self.dict_key_type):
            dict.__setitem__(self, key, value)
        else:
            self._set_data_func(self, key, value)

    def _set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self._set_data_func(x.get(new_data_key), index, new_data_value)
        else:
            if torch.is_tensor(x):
                if isinstance(new_data, torch.Tensor):
                    t = new_data
                elif isinstance(new_data, np.ndarray):
                    t = torch.from_numpy(new_data)
                else:
                    raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

                x[index].copy_(t)

            elif isinstance(x, np.ndarray):
                if isinstance(new_data, torch.Tensor):
                    n = new_data.cpu().numpy()
                elif isinstance(new_data, np.ndarray):
                    n = new_data
                else:
                    raise ValueError(f"Type {type(new_data)} not supported in set_data_func")

                x[index] = n


def clone_tensordict(d: TensorDict) -> TensorDict:
    """Returns a cloned tensordict."""
    d_clone = copy_dict_structure(d)
    for d1, d2, key, v1, v2 in iter_dicts_recursively(d, d_clone):
        d2[key] = v1.clone().detach()
    return d_clone


def shallow_recursive_copy(d: TensorDict) -> TensorDict:
    """
    Returns a shallow copy of the tensordict. Different dictionary object (recursively) but referencing
    the same tensors.
    """
    d_copy = copy_dict_structure(d)
    for d1, d2, key, v1, v2 in iter_dicts_recursively(d, d_copy):
        d2[key] = v1
    return d_copy


def tensor_dict_to_numpy(d: TensorDict) -> TensorDict:
    numpy_dict = copy_dict_structure(d)
    for d1, d2, key, curr_t, value2 in iter_dicts_recursively(d, numpy_dict):
        assert isinstance(curr_t, torch.Tensor)
        assert value2 is None
        d2[key] = curr_t.numpy()
        assert isinstance(d2[key], np.ndarray)
    return numpy_dict


def to_numpy(t: Tensor | TensorDict) -> Tensor | TensorDict:
    if isinstance(t, TensorDict):
        return tensor_dict_to_numpy(t)
    else:
        return t.numpy()  # only going to work for cpu tensors


def cat_tensordicts(lst: List[TensorDict]) -> TensorDict:
    """
    Concatenates a list of tensordicts.
    """
    if not lst:
        return TensorDict()

    res = list_of_dicts_to_dict_of_lists(lst)
    # iterate res recursively and concatenate tensors
    for d, k, v in iterate_recursively(res):
        if isinstance(v[0], torch.Tensor):
            d[k] = torch.cat(v)
        elif isinstance(v[0], np.ndarray):
            d[k] = np.concatenate(v)
        else:
            raise ValueError(f"Type {type(v[0])} not supported in cat_tensordicts")

    return TensorDict(res)


def find_invalid_data(
    t: TensorDict, msg: Optional[str] = None, keys: Optional[Iterable[str]] = None
) -> Optional[Dict[str, Tensor]]:
    res = {}
    msg = msg or "Check"

    for d, k, v in iterate_recursively(t):
        if keys is not None and k not in keys:
            continue

        if isinstance(v, torch.Tensor):
            invalid_idx = None
            if torch.is_floating_point(v):
                # check if there are any NaNs or infs
                if torch.isnan(v).any() or torch.isinf(v).any():
                    log.error(f"{msg}: Found NaNs or infs in {k}: {v}")
                    res[k] = torch.isnan(v) | torch.isinf(v)

                # noinspection PyUnresolvedReferences
                invalid_idx = (v == MAGIC_FLOAT).nonzero()
            elif torch.dtype in (torch.int, torch.int32, torch.int64, torch.int8, torch.uint8):
                # noinspection PyUnresolvedReferences
                invalid_idx = (v == MAGIC_INT).nonzero()

            if invalid_idx is not None and invalid_idx.numel() > 0:
                res[k] = invalid_idx
                log.error(f"{msg}: Found invalid data in {k} at {invalid_idx} (numel={invalid_idx.numel()})")

    return res
