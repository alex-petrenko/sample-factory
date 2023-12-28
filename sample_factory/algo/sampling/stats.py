"""
Some reusable code for stats collection.
StatsObservers are usually "Runner" objects, but they can be anything that can accumulate stats,
see evaluation_sampling_api.py for an example of a custom stats observer.
"""

from collections import deque
from typing import Any

from sample_factory.algo.utils.misc import SAMPLES_COLLECTED
from sample_factory.utils.typing import PolicyID


def timing_msg_handler(stats_observer: Any, msg: dict) -> None:
    """We use duck typing here, assuming that stats_observer object has avg_stats dict."""
    assert hasattr(stats_observer, "avg_stats"), f"stats_observer object has no avg_stats dict: {stats_observer}"

    for k, v in msg["timing"].items():
        if k not in stats_observer.avg_stats:
            stats_observer.avg_stats[k] = deque([], maxlen=50)
        stats_observer.avg_stats[k].append(v)


def stats_msg_handler(stats_observer: Any, msg: dict) -> None:
    """We use duck typing here, assuming that stats_observer object has stats dict."""
    assert hasattr(stats_observer, "stats"), f"stats_observer object has no stats dict: {stats_observer}"
    stats_observer.stats.update(msg["stats"])


def samples_stats_handler(stats_observer: Any, msg: dict, policy_id: PolicyID) -> None:
    stats_observer.samples_collected[policy_id] += msg[SAMPLES_COLLECTED]
