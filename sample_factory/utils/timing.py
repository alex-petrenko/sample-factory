import collections
import dataclasses
import time
import typing
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Union

import psutil

from sample_factory.algo.utils.misc import EPS
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log


class AvgTime:
    def __init__(self, num_values_to_avg):
        self.values = deque([], maxlen=num_values_to_avg)

    def __str__(self):
        avg_time = sum(self.values) / max(1, len(self.values))
        return f"{avg_time:.4f}"


@dataclass
class TimingTreeNode:
    self_time: Union[float, AvgTime] = 0
    timing: typing.OrderedDict[str, Any] = dataclasses.field(default_factory=collections.OrderedDict)


# noinspection PyProtectedMember
class TimingContext:
    def __init__(self, timing, key: str, additive=False, average=None):
        super().__init__()
        self.timing_tree_node: Optional[TimingTreeNode] = None

        self._timing = timing

        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None
        self._time = 0

    def set_tree_node(self, node):
        self.timing_tree_node = node

    def initial_value(self):
        if self._average is not None:
            return AvgTime(num_values_to_avg=self._average)
        return 0.0

    def _record_measurement(self, key, value):
        if self._additive:
            self._timing[key] += value
            self.timing_tree_node.self_time += value
        elif self._average is not None:
            self._timing[key].values.append(value)
            self.timing_tree_node.self_time.values.append(value)
        else:
            self._timing[key] = value
            self.timing_tree_node.self_time = value

    def __enter__(self):
        self._time_enter = time.time()
        self._timing._open_contexts_stack.append(self)

    def __exit__(self, type_, value, traceback):
        time_passed = max(time.time() - self._time_enter, EPS)  # EPS to prevent div by zero
        self._record_measurement(self._key, time_passed)
        self._timing._open_contexts_stack.pop()


class Timing(AttrDict):
    def __init__(self, name="Profile", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._name = name

        self._root_context = TimingContext(self, "~")
        self._root_context.set_tree_node(TimingTreeNode())
        self._open_contexts_stack = [self._root_context]

    def _push_context(self, ctx):
        self._open_contexts_stack.append(ctx)
        return ctx

    def _init_context(self, key, *args, **kwargs):
        ctx = TimingContext(self, key, *args, **kwargs)
        if key not in self:
            self[key] = ctx.initial_value()

        parent_ctx = self._open_contexts_stack[-1]
        parent_tree_node = parent_ctx.timing_tree_node
        if key not in parent_tree_node.timing:
            parent_tree_node.timing[key] = TimingTreeNode(ctx.initial_value())

        ctx.set_tree_node(parent_tree_node.timing[key])
        return ctx

    def timeit(self, key):
        return self._init_context(key)

    def add_time(self, key):
        return self._init_context(key, additive=True)

    def time_avg(self, key, average=10):
        return self._init_context(key, average=average)

    @staticmethod
    def _time_str(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    def flat_str(self):
        # skip data members of Timing
        skip_names = ["_root_context", "_open_contexts_stack"]

        s = []
        for key, value in self.items():
            if key not in skip_names:
                s.append(f"{key}: {self._time_str(value)}")
        return ", ".join(s)

    @classmethod
    def _tree_str_func(cls, node: TimingTreeNode, depth: int):
        indent = " " * 2 * depth

        leaf_nodes = ((k, v) for k, v in node.timing.items() if not v.timing)
        nonleaf_nodes = ((k, v) for k, v in node.timing.items() if v.timing)

        def node_str(k, node_):
            return f"{k}: {cls._time_str(node_.self_time)}"

        tokens = []
        for key, child_node in leaf_nodes:
            tokens.append(node_str(key, child_node))

        lines = []
        if tokens:
            lines.append(f'{indent}{", ".join(tokens)}')

        for key, child_node in nonleaf_nodes:
            lines.append(f"{indent}{node_str(key, child_node)}")
            lines.extend(cls._tree_str_func(child_node, depth + 1))

        return lines

    def tree_str(self):
        lines = [f"{self._name} tree view:"]
        lines.extend(self._tree_str_func(self._root_context.timing_tree_node, 0))
        return "\n".join(lines)

    def __str__(self):
        return self.tree_str()


def init_global_profiler(t):
    """This is for debugging purposes. Normally prefer to pass it around."""
    global TIMING
    log.info("Setting global profiler in process %r", psutil.Process())
    TIMING = t
