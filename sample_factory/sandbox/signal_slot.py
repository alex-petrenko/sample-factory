from __future__ import annotations

import multiprocessing
import sys
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from queue import Empty
from typing import Dict, Any, Set, Callable

from sample_factory.algo.utils.queues import get_mp_queue
from sample_factory.utils.utils import log

# Considerations:
# - Event loop should handle timers!!


# Implement:
# - connect slots with signals
# - event loop in main process and in other processes
# - multiple objects living on the same event loop exchanging messages
# - connecting objects from different processes (??)
# - blocking events (??) i.e. save model and wait for it to be saved


class EventLoopObject:
    all_objects: Dict[str, EventLoopObject] = dict()

    def __init__(self, event_loop, object_id=None):
        # the reason we can't use regular id() is because depending on the process spawn method the same objects
        # can have the same id() (in Fork method) or different id() (spawn method)
        self.object_id = object_id if object_id is not None else str(uuid.uuid4())
        assert self.object_id not in self.all_objects

        self.event_loop: EventLoop = event_loop

        # receivers of signals emitted by this object
        self.receivers: Dict[str, Set[Receiver]] = dict()
        self.receiver_queues: Dict[str, Dict[Any, int]] = dict()

        # emitters we receive signals from
        self.emitters: Set[Emitter] = set()

        self.all_objects[self.object_id] = self

    @staticmethod
    def _add_to_dict_of_sets(d: Dict[Any, Set], key, value):
        if key not in d:
            d[key] = set()
        d[key].add(value)

    def connect(self, signal: str, other_object: EventLoopObject, slot: str):
        emitter = Emitter(self.object_id, signal)
        receiver = Receiver(other_object.object_id, slot)

        self._add_to_dict_of_sets(self.receivers, signal, receiver)

        receiving_loop = other_object.event_loop
        self.receiver_event_loops[receiving_loop] = self.receiver_event_loops.get(receiving_loop, 0) + 1

        self._add_to_dict_of_sets(receiving_loop.receivers, emitter, receiver)

        self.emitters.add(emitter)

    def disconnect(self, signal, other_object: EventLoopObject, slot: str):
        if signal not in self.receivers:
            log.warning(f'{self.object_id}:{signal=} is not connected to anything')
            return

        receiver = Receiver(other_object.object_id, slot)
        if receiver not in self.receivers[signal]:
            log.warning(f'{self.object_id}:{signal=} is not connected to {other_object.object_id}:{slot=}')
            return

        self.receivers[signal].remove(receiver)

        receiving_loop = other_object.event_loop
        if self.receiver_event_loops.get(receiving_loop, 0) == 0:
            del self.receiver_event_loops[receiving_loop]

        other_object.emitters.remove(self.object_id)

    def emit(self, signal_name: str, data: Any = None):
        for receiver in self.receiver_event_loops[signal_name]:
            # we just push one message into each receiver event loop queue
            # event loops themselves will redistribute the signal to all receivers living on that loop
            receiver.signal_queue.put((self.object_id, signal_name, data), block=True, timeout=1.0)

    def move_to_loop(self, loop: EventLoop):
        assert isinstance(loop, EventLoop), f'Should use an instance of {EventLoop.__name__}'

        # remove from the existing loop receivers

        self.event_loop = loop



    def __del__(self):
        # TODO disconnect everything
        log.debug(f'__del__ called for {self.object_id}')
        del self.all_objects[self.object_id]
        pass


def connect(obj1: EventLoopObject, signal: str, obj2: EventLoopObject, slot: str):
    obj1.connect(signal, obj2, slot)


@dataclass(frozen=True)
class Emitter:
    object_id: Any
    signal_name: str


@dataclass(frozen=True)
class Receiver:
    object_id: Any
    slot_name: str


class EventLoop:
    def __init__(self, loop_frequency_hz=50):
        self.signal_queue = get_mp_queue()
        self.loop_timeout_sec = 1.0 / loop_frequency_hz

        self.objects: Set[EventLoopObject] = set()
        self.receivers: Dict[Emitter, Set[Receiver]] = dict()

        self.should_terminate = False

    def terminate(self):
        self.should_terminate = True

    def _process_signal(self, signal):
        emitter_object_id, signal_name, data = signal
        emitter = Emitter(emitter_object_id, signal_name)

        receivers = self.receivers.get(emitter, ())

        for receiver in receivers:
            obj_id, slot = receiver.object_id, receiver.slot_name
            obj = EventLoopObject.all_objects.get(receiver.object_id)
            if obj is None:
                log.warning(f'Attempting to call a slot on object {obj_id} which is not found')
                continue

            if not hasattr(obj, slot):
                log.warning(f'{slot=} not found in object {obj_id}')
                continue

            slot_callable = getattr(obj, slot)
            if not isinstance(slot_callable, Callable):
                log.warning(f'{slot=} of {obj_id=} is not callable')
                continue

            slot_callable(data)

    def exec(self):
        while not self.should_terminate:
            try:
                # loop over all incoming signals, see if any of the objects living on this event loop are connected
                # to this particular signal, call slots if needed
                signals = self.signal_queue.get_many(timeout=self.loop_timeout_sec)
            except Empty:
                signals = ()

            for signal in signals:
                self._process_signal(signal)


class EventLoopProcess(multiprocessing.Process):
    def __init__(self, name=None, daemon=None):
        super().__init__(target=self._target, name=name, daemon=daemon)
        self.event_loop = EventLoop()

    def _target(self):
        self.event_loop.exec()


class C1(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.x = 42

    def inc(self, data):
        self.x += 1
        print(f'inc slot {data=}')


class C2(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.pi = 3.14

    def foo(self):
        print('Foo')
        self.emit('foo_signal', 123)


def main():
    # create some objects, connect them, run the event loop
    event_loop = EventLoop()

    o1 = C1(event_loop, 'o1')
    o2 = C2(event_loop, 'o2')

    connect(o2, 'foo_signal', o1, 'inc')

    o2.foo()

    p = EventLoopProcess()
    o1.move_to_loop(p)
    p.start()

    event_loop.exec()

    p.join()


if __name__ == '__main__':
    sys.exit(main())
