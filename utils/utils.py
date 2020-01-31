"""Utilities."""
import argparse
import logging
import operator
import os
from os.path import join

import numpy as np
import psutil
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)


# general utilities

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


def scale_to_range(np_array, min_, max_):
    min_arr = np.min(np_array)
    max_arr = np.max(np_array)
    ret_array = (np_array - min_arr) / (max_arr - min_arr)  # scale to (0,1)

    ret_array = ret_array * (max_ - min_) + min_  # scale to (min, max)
    return ret_array


def op_with_idx(x, op):
    assert len(x) > 0

    best_idx = 0
    best_x = x[best_idx]
    for i, item in enumerate(x):
        if op(item, best_x):
            best_x = item
            best_idx = i

    return best_x, best_idx


def min_with_idx(x):
    return op_with_idx(x, operator.lt)


def max_with_idx(x):
    return op_with_idx(x, operator.gt)


# CLI args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', ):
        return True
    elif isinstance(v, str) and v.lower() in ('false', ):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


# numpy stuff

def numpy_all_the_way(list_of_arrays):
    """Turn a list of numpy arrays into a 2D numpy array."""
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr


def numpy_flatten(list_of_arrays):
    """Turn a list of numpy arrays into a 1D numpy array (flattened)."""
    return np.concatenate(list_of_arrays, axis=0)


def ensure_contigious(x):
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


# matplotlib

def figure_to_numpy(figure):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param figure a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    figure.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = figure.canvas.get_width_height()
    buffer = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
    buffer.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buffer = np.roll(buffer, 3, axis=2)
    return buffer


# os-related stuff

def memory_consumption_mb():
    """Memory consumption of the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


def join_or_kill(process, timeout=1.0):
    process.join(timeout)
    if process.is_alive():
        log.warning('Process %r could not join, kill it with fire!', process)
        process.kill()
        log.warning('Process %r is dead (%r)', process, process.is_alive())


# working with filesystem

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def remove_if_exists(file):
    if os.path.isfile(file):
        os.remove(file)


def project_root():
    """
    Keep models, parameters and summaries at the root of this project's directory tree.
    :return: full path to the root dir of this project.
    """
    return os.path.dirname(os.path.dirname(__file__))


def experiments_dir():
    return ensure_dir_exists(join(project_root(), 'train_dir'))


def experiment_dir(experiment=None, experiments_root=None, cfg=None):
    if cfg is not None:
        experiment = cfg.experiment
        experiments_root = cfg.experiments_root

    if experiments_root is None:
        experiments_root = experiments_dir()
    else:
        experiments_root = join(experiments_dir(), experiments_root)

    return ensure_dir_exists(join(experiments_root, experiment))


def model_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.model'))


def summaries_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.summary'))


def cfg_file(cfg):
    params_file = join(experiment_dir(cfg=cfg), 'cfg.json')
    return params_file
