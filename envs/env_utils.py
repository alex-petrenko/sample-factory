from utils.utils import is_module_available, log
from functools import wraps
from time import sleep


class EnvCriticalError(Exception):
    pass


def vizdoom_available():
    return is_module_available('vizdoom')


def minigrid_available():
    return is_module_available('gym_minigrid')


def quadrotors_available():
    return is_module_available('gym_art')


def dmlab_available():
    return is_module_available('deepmind_lab')


def voxel_env_available():
    return is_module_available('voxel_env')


def retry(exception_class=Exception, num_attempts=3, sleep_time=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    if i == num_attempts - 1:
                        raise
                    else:
                        log.error('Failed with error %r, trying again', e)
                        sleep(sleep_time)
        return wrapper
    return decorator
