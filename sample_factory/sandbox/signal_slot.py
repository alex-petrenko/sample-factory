from __future__ import annotations

import multiprocessing
import sys
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from queue import Empty
from typing import Dict, Any, Set, Callable, Tuple

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
    _obj_ids: Set[Any] = set()

    def __init__(self, event_loop, object_id=None):
        # the reason we can't use regular id() is because depending on the process spawn method the same objects
        # can have the same id() (in Fork method) or different id() (spawn method)
        self.object_id = object_id if object_id is not None else str(uuid.uuid4())
        assert self.object_id not in self._obj_ids, f'{self.object_id=} is not unique!'

        self._obj_ids.add(object_id)

        self.event_loop: EventLoop = event_loop
        self.event_loop.objects[self.object_id] = self

        # receivers of signals emitted by this object
        self.send_signals_to: Dict[str, Set[Any]] = dict()
        self.receiver_queues: Dict = dict()
        self.receiver_refcount: Dict[Any, int] = dict()

        # connections (emitter -> slot name)
        self.connections: Dict[Emitter, str] = dict()

    def _add_to_loop(self, loop):
        self.event_loop = loop
        self.event_loop.objects[self.object_id] = self

    @staticmethod
    def _add_to_dict_of_sets(d: Dict[Any, Set], key, value):
        if key not in d:
            d[key] = set()
        d[key].add(value)

    def connect(self, signal: str, other: EventLoopObject, slot: str):
        emitter = Emitter(self.object_id, signal)
        receiver_id = other.object_id

        self._add_to_dict_of_sets(self.send_signals_to, signal, receiver_id)

        receiving_loop = other.event_loop
        self._add_to_dict_of_sets(receiving_loop.receivers, emitter, receiver_id)

        q = receiving_loop.signal_queue
        self.receiver_queues[receiver_id] = q
        self.receiver_refcount[receiver_id] = self.receiver_refcount.get(receiver_id, 0) + 1

        other.connections[emitter] = slot

    def disconnect(self, signal, other_object: EventLoopObject, slot: str):
        if signal not in self.send_signals_to:
            log.warning(f'{self.object_id}:{signal=} is not connected to anything')
            return

        receiver_id = other_object.object_id
        if receiver_id not in self.send_signals_to[signal]:
            log.warning(f'{self.object_id}:{signal=} is not connected to {receiver_id}:{slot=}')
            return

        self.send_signals_to[signal].remove(receiver_id)

        self.receiver_refcount[receiver_id] -= 1
        if self.receiver_refcount[receiver_id] <= 0:
            del self.receiver_refcount[receiver_id]
            del self.receiver_queues[receiver_id]

        emitter = Emitter(self.object_id, signal)
        del other_object.connections[emitter]

        loop_receivers = other_object.event_loop.receivers.get(emitter)
        if loop_receivers is not None:
            loop_receivers.remove(other_object.object_id)

    def emit(self, signal_name: str, data: Any = None):
        # find a set of queues we need to send this signal to
        queues = set()
        for receiver_id in self.send_signals_to[signal_name]:
            queues.add(self.receiver_queues[receiver_id])

        for q in queues:
            # we just push one message into each receiver event loop queue
            # event loops themselves will redistribute the signal to all receivers living on that loop
            q.put((self.object_id, signal_name, data), block=True, timeout=1.0)

    def move_to_loop(self, loop: EventLoop):
        assert isinstance(loop, EventLoop), f'Should use an instance of {EventLoop.__name__}'

        # remove from the existing loop receivers
        del self.event_loop.objects[self.object_id]

        # remove mentions of this object from the current loop
        # since we're moving to a different loop, we should not receive signals on this loop anymore
        for emitter in self.connections:
            self.event_loop.receivers[emitter].remove()

        # TODO

        self._add_to_loop(loop)

    def __del__(self):
        # TODO disconnect everything
        log.debug(f'__del__ called for {self.object_id}')
        self._obj_ids.remove(self.object_id)
        pass


def connect(obj1: EventLoopObject, signal: str, obj2: EventLoopObject, slot: str):
    obj1.connect(signal, obj2, slot)


def disconnect(obj1: EventLoopObject, signal: str, obj2: EventLoopObject, slot: str):
    obj1.disconnect(signal, obj2, slot)


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

        # objects living on this loop
        self.objects: Dict[Any, EventLoopObject] = dict()

        self.receivers: Dict[Emitter, Set[Any]] = dict()

        self.should_terminate = False

    def terminate(self):
        self.should_terminate = True

    def _process_signal(self, signal):
        emitter_object_id, signal_name, data = signal
        emitter = Emitter(emitter_object_id, signal_name)

        receiver_ids = self.receivers.get(emitter, ())

        for obj_id in receiver_ids:
            obj = self.objects.get(obj_id)
            if obj is None:
                log.warning(f'Attempting to call a slot on object {obj_id} which is not found on this loop')
                continue

            slot = obj.connections.get(emitter)
            if obj is None:
                log.warning(f'{emitter=} does not appear to be connected to {obj_id=}')
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
        try:
            while not self.should_terminate:
                try:
                    # loop over all incoming signals, see if any of the objects living on this event loop are connected
                    # to this particular signal, call slots if needed
                    signals = self.signal_queue.get_many(timeout=self.loop_timeout_sec)
                except Empty:
                    signals = ()

                for signal in signals:
                    self._process_signal(signal)
        except KeyboardInterrupt:
            log.warning(f'Keyboard interrupt detected in the event loop {self}, exiting...')


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
    disconnect(o2, 'foo_signal', o1, 'inc')
    connect(o2, 'foo_signal', o1, 'inc')

    o2.foo()

    p = EventLoopProcess()
    p.start()

    event_loop.exec()

    p.join()


if __name__ == '__main__':
    sys.exit(main())
