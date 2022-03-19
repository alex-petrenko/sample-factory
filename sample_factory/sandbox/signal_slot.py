import sys
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Set

from sample_factory.algo.utils.queues import get_mp_queue

# Implement:
# - connect slots with signals
# - event loop in main process and in other processes
# - multiple objects living on the same event loop exchanging messages
# - connecting objects from different processes (??)
# - blocking events (??) i.e. save model and wait for it to be saved

# Considerations:
# -


class EventLoopObject:
    pass


@dataclass
class SignalDescription:
    object_id: Any
    signal_name: str


@dataclass
class Connection:
    slot: Bo


class EventLoop:
    def __init__(self, loop_frequency_hz=50):
        self.signal_queue = get_mp_queue()
        self.loop_timeout_sec = 1.0 / loop_frequency_hz

        self.connections: Dict[SignalDescription, Set[Connection]] = dict()

    def run(self):
        while True:  # TODO: termination
            # loop over all incoming signals, see if any of the objects living on this event loop are connected
            # to this particular signal, call slots if needed
            signals = self.signal_queue.get_many(timeout=self.loop_timeout_sec)

            for signal in signals:
                object_id, signal_name, data = signal
                connected_objects = self.connections.get((object_id, signal_name), ())

                for connected_object in connected_objects:




def main():
    pass


if __name__ == '__main__':
    sys.exit(main())
