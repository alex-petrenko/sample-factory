"""Utilities."""

import argparse
import importlib
import logging
import operator
import os
import pwd
import tempfile
from _queue import Empty
from os.path import join
from subprocess import check_output, CalledProcessError, run
from sys import platform

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


# general Python utilities

def is_module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


def set_attr_if_exists(obj, attr_name, attr_value):
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, attr_value)


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


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.info('Queue timed out (%s), timeout %.3f', msg, timeout)


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

def get_free_disk_space_mb(cfg):
    statvfs = os.statvfs(experiments_dir(cfg))
    return statvfs.f_frsize * statvfs.f_bfree / (1024 * 1024)


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


def list_child_processes():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    is_alive = []
    for child in children:
        try:
            child_process = psutil.Process(child.pid)
            if child_process.is_running():
                is_alive.append(child_process)
        except psutil.NoSuchProcess:
            pass

    return is_alive


def kill_processes(processes):
    for p in processes:
        try:
            if 'torch_shm' in p.name():
                # do not kill to avoid permanent memleaks
                # https://pytorch.org/docs/stable/multiprocessing.html#file-system-file-system
                continue

            # log.debug('Child process name %d %r %r %r', p.pid, p.name(), p.exe(), p.cmdline())
            log.debug('Child process name %d %r %r', p.pid, p.name(), p.exe())
            if p.is_running():
                log.debug('Killing process %s...', p.name())
                p.kill()
        except psutil.NoSuchProcess:
            # log.debug('Process %d is already dead', p.pid)
            pass


def cores_for_worker_process(worker_idx, num_workers, cpu_count):
    worker_idx_modulo = worker_idx % cpu_count

    # trying to optimally assign workers to CPU cores to minimize context switching
    # logic here is best illustrated with an example
    # 20 cores, 44 workers (why? I don't know, someone wanted 44 workers)
    # first 40 are assigned to a single core each, remaining 4 get 5 cores each

    cores = None
    whole_workers_per_core = num_workers // cpu_count
    if worker_idx < whole_workers_per_core * cpu_count:
        # these workers get an private core each
        cores = [worker_idx_modulo]
    else:
        # we're dealing with some number of workers that is less than # of cpu cores
        remaining_workers = num_workers % cpu_count
        if cpu_count % remaining_workers == 0:
            cores_to_use = cpu_count // remaining_workers
            cores = list(range(worker_idx_modulo * cores_to_use, (worker_idx_modulo + 1) * cores_to_use, 1))

    return cores


def set_process_cpu_affinity(worker_idx, num_workers):
    if platform == 'darwin':
        log.debug('On MacOS, not setting affinity')
        return

    curr_process = psutil.Process()
    cpu_count = psutil.cpu_count()
    cores = cores_for_worker_process(worker_idx, num_workers, cpu_count)
    if cores is not None:
        curr_process.cpu_affinity(cores)

    log.debug('Worker %d uses CPU cores %r', worker_idx, curr_process.cpu_affinity())


# working with filesystem

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_ensure_dir_exists(path):
    """Should be safer in multi-treaded environment."""
    try:
        return ensure_dir_exists(path)
    except FileExistsError:
        return path


def remove_if_exists(file):
    if os.path.isfile(file):
        os.remove(file)


def get_username():
    return pwd.getpwuid(os.getuid()).pw_name


def project_tmp_dir():
    tmp_dir_name = f'sample_factory_{get_username()}'
    return ensure_dir_exists(join(tempfile.gettempdir(), tmp_dir_name))


def experiments_dir(cfg):
    return ensure_dir_exists(cfg.train_dir)


def experiment_dir(cfg):
    experiment = cfg.experiment
    experiments_root = cfg.experiments_root

    if experiments_root is None:
        experiments_root = experiments_dir(cfg)
    else:
        experiments_root = join(experiments_dir(cfg), experiments_root)

    return ensure_dir_exists(join(experiments_root, experiment))


def model_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.model'))


def summaries_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.summary'))


def cfg_file(cfg):
    params_file = join(experiment_dir(cfg=cfg), 'cfg.json')
    return params_file


def done_filename(cfg):
    return join(experiment_dir(cfg=cfg), 'done')


def get_git_commit_hash():
    path_to_project = os.path.dirname(os.path.realpath(__file__))
    try:
        git_hash = check_output(['git', 'rev-parse', 'HEAD'],
                                cwd=path_to_project,
                                timeout=5).strip().decode('ascii')
    except CalledProcessError:
        # this scenario is for when there's no git and we are returning an unknown value
        git_hash = 'unknown'
    return git_hash


def save_git_diff(directory):
    path_to_project = os.path.dirname(os.path.realpath(__file__))
    try:
        with open(join(directory, 'git.diff'), 'w') as outfile:
            run(['git', 'diff'],
                stdout=outfile, cwd=path_to_project, timeout=5)
    except CalledProcessError:
        pass
