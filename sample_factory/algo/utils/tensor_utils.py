from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor


def clone_tensor(t: Tensor | np.ndarray) -> Tensor | np.ndarray:
    if isinstance(t, Tensor):
        return t.clone().detach()
    else:
        return np.copy(t)


def unsqueeze_tensor(t: Tensor | np.ndarray, dim: int) -> Tensor | np.ndarray:
    if isinstance(t, Tensor):
        return t.unsqueeze(dim)
    else:
        return np.expand_dims(t, dim)


def cat_tensors(lt: List[Tensor | np.ndarray]) -> Tensor | np.ndarray:
    if isinstance(lt[0], Tensor):
        return torch.cat(lt)
    else:
        return np.concatenate(lt)


def dict_of_lists_cat(d: Dict[Any, List | Tensor]):
    for key, x in d.items():
        d[key] = cat_tensors(x)


def ensure_torch_tensor(t: Tensor | np.ndarray) -> Tensor:
    if isinstance(t, Tensor):
        return t
    else:
        return torch.from_numpy(t)


def ensure_numpy_array(t: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(t, Tensor):
        return t.numpy()
    else:
        return t
