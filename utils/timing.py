import time

from algorithms.utils.algo_utils import EPS
from utils.utils import AttrDict


class TimingContext:
    def __init__(self, timer, key, additive=False):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._time_enter = None

    def __enter__(self):
        self._time_enter = time.time()

    def __exit__(self, type_, value, traceback):
        if self._key not in self._timer:
            self._timer[self._key] = 0

        time_passed = max(time.time() - self._time_enter, EPS)  # EPS to prevent div by zero

        if self._additive:
            self._timer[self._key] += time_passed
        else:
            self._timer[self._key] = time_passed


class Timing(AttrDict):
    def __init__(self, d=None):
        super(Timing, self).__init__(d)

    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def __str__(self):
        s = ''
        for key, value in self.items():
            s += f'{key}: {value:.3f}, '
        return s
