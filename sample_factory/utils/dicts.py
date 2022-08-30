from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, OrderedDict


def dict_of_lists_append(d: Dict[Any, List], new_data):
    for key, x in new_data.items():
        if key in d:
            d[key].append(x)
        else:
            d[key] = [x]


def dict_of_lists_append_idx(d: Dict[Any, List], new_data, index):
    for key, x in new_data.items():
        if key in d:
            d[key].append(x[index])
        else:
            d[key] = [x[index]]


def iterate_recursively(d):
    """
    Generator for a dictionary that can potentially include other dictionaries.
    Yields tuples of (dict, key, value), where key, value are "leaf" elements of the "dict".

    """
    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v


def iterate_recursively_with_prefix(d: Dict, prefix=None):
    """
    Generator for a dictionary that can potentially include other dictionaries.
    Yields tuples of (dict, key, value, prefix), where key, value are "leaf" elements of the "dict" and prefix is a
    list of keys that lead to the current element (exluding the current key).

    """
    if prefix is None:
        prefix = []

    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively_with_prefix(v, prefix + [k])
        else:
            yield d, k, v, prefix


def copy_dict_structure(d):
    """Copy dictionary layout without copying the actual values (populated with Nones)."""
    d_copy = type(d)()
    _copy_dict_structure_func(d, d_copy)
    return d_copy


def _copy_dict_structure_func(d, d_copy):
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            d_copy[key] = type(value)()
            _copy_dict_structure_func(value, d_copy[key])
        else:
            d_copy[key] = None


def iter_dicts_recursively(d1, d2):
    """
    Assuming structure of d1 is strictly included into d2.
    I.e. each key at each recursion level is also present in d2. This is also true when d1 and d2 have the same
    structure.
    """
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


def list_of_dicts_to_dict_of_lists(list_of_dicts: List[Dict]) -> Dict[Any, List]:
    if not list_of_dicts:
        return dict()

    res = copy_dict_structure(list_of_dicts[0])

    for d in list_of_dicts:
        for d1, d2, key, v1, v2 in iter_dicts_recursively(d, res):
            if v2 is None:
                d2[key] = [v1]
            else:
                d2[key].append(v1)

    return res


def get_first_present(d: Dict, keys: Iterable, default: Optional[Any] = None) -> Optional[Any]:
    for key in keys:
        if key in d:
            return d[key]
    return default
