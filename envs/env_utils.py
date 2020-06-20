from utils.utils import is_module_available


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
