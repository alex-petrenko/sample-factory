from __future__ import annotations

from typing import Dict, Any, List, OrderedDict


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


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = dict()

    for d in list_of_dicts:
        for key, x in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []

            dict_of_lists[key].append(x)

    return dict_of_lists
