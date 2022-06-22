from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from sample_factory.utils.dicts import copy_dict_structure, iter_dicts_recursively


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
                    raise Exception(f'Type {type(new_data)} not supported in set_data_func')

                x[index].copy_(t)

            elif isinstance(x, np.ndarray):
                if isinstance(new_data, torch.Tensor):
                    n = new_data.cpu().numpy()
                elif isinstance(new_data, np.ndarray):
                    n = new_data
                else:
                    raise Exception(f'Type {type(new_data)} not supported in set_data_func')

                x[index] = n


def clone_tensordict(d: TensorDict) -> TensorDict:
    """Returns a cloned tensordict."""
    d_clone = copy_dict_structure(d)
    for d1, d2, key, v1, v2 in iter_dicts_recursively(d, d_clone):
        d2[key] = v1.clone().detach()
    return d_clone


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
