from __future__ import annotations

import multiprocessing
import os
import time
import types
import uuid
from dataclasses import dataclass
from queue import Empty
from typing import Dict, Any, Set, Callable, Union, List, Optional

import psutil

from sample_factory.algo.utils.queues import get_mp_queue
from sample_factory.utils.utils import log

# type aliases for clarity
ObjectID = Any  # ObjectID can be any hashable type, usually a string
MpQueue = Any  # can actually be any multiprocessing Queue type, i.e. faster_fifo queue
BoundMethod = Any
StatusCode = int


@dataclass(frozen=True)
class Emitter:
    object_id: ObjectID
    signal_name: str


@dataclass(frozen=True)
class Receiver:
    object_id: ObjectID
    slot_name: str


class EventLoopObject:
    _obj_ids: Set[ObjectID] = set()

    def __init__(self, event_loop, object_id=None):
        # the reason we can't use regular id() is because depending on the process spawn method the same objects
        # can have the same id() (in Fork method) or different id() (spawn method)
        self.object_id = object_id if object_id is not None else self._default_obj_id()
        assert self.object_id not in self._obj_ids, f'{self.object_id=} is not unique!'

        self._obj_ids.add(self.object_id)

        self.event_loop: EventLoop = event_loop
        self.event_loop.objects[self.object_id] = self

        # receivers of signals emitted by this object
        self.send_signals_to: Dict[str, Set[ObjectID]] = dict()
        self.receiver_queues: Dict[ObjectID, MpQueue] = dict()
        self.receiver_refcount: Dict[ObjectID, int] = dict()

        # connections (emitter -> slot name)
        self.connections: Dict[Emitter, str] = dict()

    def _default_obj_id(self):
        return str(uuid.uuid4())

    def _add_to_loop(self, loop):
        self.event_loop = loop
        self.event_loop.objects[self.object_id] = self

    @staticmethod
    def _add_to_dict_of_sets(d: Dict[Any, Set], key, value):
        if key not in d:
            d[key] = set()
        d[key].add(value)

    @staticmethod
    def _throw_if_different_processes(o1: EventLoopObject, o2: EventLoopObject):
        o1_p, o2_p = o1.event_loop.process, o2.event_loop.process
        if o1_p != o2_p:
            msg = f'Objects {o1.object_id} and {o2.object_id} live on different processes'
            log.error(msg)
            raise RuntimeError(msg)

    @staticmethod
    def _bound_method_to_obj_slot(obj, slot):
        if isinstance(obj, (types.MethodType, types.BuiltinMethodType)):
            slot = obj.__name__
            obj = obj.__self__

        assert isinstance(obj, EventLoopObject)
        assert slot is not None
        return obj, slot

    def connect(self, signal: str, other: Union[EventLoopObject, BoundMethod], slot: str = None):
        other, slot = self._bound_method_to_obj_slot(other, slot)

        self._throw_if_different_processes(self, other)

        emitter = Emitter(self.object_id, signal)
        receiver_id = other.object_id

        self._add_to_dict_of_sets(self.send_signals_to, signal, receiver_id)

        receiving_loop = other.event_loop
        self._add_to_dict_of_sets(receiving_loop.receivers, emitter, receiver_id)

        q = receiving_loop.signal_queue
        self.receiver_queues[receiver_id] = q
        self.receiver_refcount[receiver_id] = self.receiver_refcount.get(receiver_id, 0) + 1

        other.connections[emitter] = slot

    def disconnect(self, signal, other: Union[EventLoopObject, BoundMethod], slot: str = None):
        other, slot = self._bound_method_to_obj_slot(other, slot)

        self._throw_if_different_processes(self, other)

        if signal not in self.send_signals_to:
            log.warning(f'{self.object_id}:{signal=} is not connected to anything')
            return

        receiver_id = other.object_id
        if receiver_id not in self.send_signals_to[signal]:
            log.warning(f'{self.object_id}:{signal=} is not connected to {receiver_id}:{slot=}')
            return

        self.send_signals_to[signal].remove(receiver_id)

        self.receiver_refcount[receiver_id] -= 1
        if self.receiver_refcount[receiver_id] <= 0:
            del self.receiver_refcount[receiver_id]
            del self.receiver_queues[receiver_id]

        emitter = Emitter(self.object_id, signal)
        del other.connections[emitter]

        loop_receivers = other.event_loop.receivers.get(emitter)
        if loop_receivers is not None:
            loop_receivers.remove(other.object_id)

    def register_broadcast(self, event_loop: EventLoop, signal: str):
        self.connect(signal, event_loop.broadcast)

    def subscribe(self, signal: str, slot: Union[BoundMethod, str]):
        if isinstance(slot, (types.MethodType, types.BuiltinMethodType)):
            slot = slot.__name__
        self.event_loop.connect(signal, self, slot)

    def unsubscribe(self, signal: str, slot: Union[BoundMethod, str]):
        if isinstance(slot, (types.MethodType, types.BuiltinMethodType)):
            slot = slot.__name__
        self.event_loop.disconnect(signal, self, slot)

    def emit(self, signal: str, *args):
        pid = self.event_loop.process.pid
        if os.getpid() != pid:
            raise RuntimeError(
                f'Cannot emit {signal}: object {self.object_id} lives on a different process {pid}!'
            )

        # find a set of queues we need to send this signal to
        queues = set()
        for receiver_id in self.send_signals_to.get(signal, ()):
            queues.add(self.receiver_queues[receiver_id])

        for q in queues:
            # we just push one message into each receiver event loop queue
            # event loops themselves will redistribute the signal to all receivers living on that loop
            q.put((self.object_id, signal, args), block=True, timeout=1.0)

    def detach(self):
        """Detach the object from it's current event loop."""
        if self.event_loop:
            del self.event_loop.objects[self.object_id]
            self.event_loop = None

    def __del__(self):
        self.detach()
        self._obj_ids.remove(self.object_id)


def connect(obj1: EventLoopObject, signal: str, other: Union[EventLoopObject, BoundMethod], slot: str = None):
    obj1.connect(signal, other, slot)


def disconnect(obj1: EventLoopObject, signal: str, other: Union[EventLoopObject, BoundMethod], slot: str = None):
    obj1.disconnect(signal, other, slot)


def subscribe(signal: str, obj: EventLoopObject, slot: Union[BoundMethod, str]):
    obj.subscribe(signal, slot)


def unsubscribe(signal: str, obj: EventLoopObject, slot: Union[BoundMethod, str]):
    obj.subscribe(signal, slot)


class EventLoopStatus:
    NORMAL_TERMINATION, INTERRUPTED = range(2)


class EventLoop(EventLoopObject):
    def __init__(self, unique_loop_name):
        # objects living on this loop
        self.objects: Dict[ObjectID, EventLoopObject] = dict()

        super().__init__(self, unique_loop_name)

        # when event loop is created it just lives on the current process
        self.process: Union[psutil.Process, EventLoopProcess] = psutil.Process(os.getpid())

        self.signal_queue = get_mp_queue()

        # Separate container to keep track of timers living on this thread. Start with one default timer.
        self.timers: List[Timer] = []
        self.default_timer = Timer(self, 0.1, object_id=f'{self.object_id}_timer')

        self.receivers: Dict[Emitter, Set[ObjectID]] = dict()

        # emitter of the signal which is currently being processed
        self.curr_emitter: Optional[Emitter] = None

        self.should_terminate = False

    def add_timer(self, t: Timer):
        self.timers.append(t)

    def remove_timer(self, t: Timer):
        self.timers.remove(t)

    def terminate(self):
        log.debug(f'Event loop {self.object_id} terminating...')
        self.should_terminate = True

    def broadcast(self, *args):
        curr_signal = self.curr_emitter.signal_name

        # we could re-emit the signal to reuse the existing signal propagation mechanism, but we can avoid
        # doing this to reduce overhead
        self._process_signal((self.object_id, curr_signal, args))

    def _process_signal(self, signal):
        emitter_object_id, signal_name, args = signal
        emitter = Emitter(emitter_object_id, signal_name)

        receiver_ids = tuple(self.receivers.get(emitter, ()))

        for obj_id in receiver_ids:
            obj = self.objects.get(obj_id)
            if obj is None:
                log.warning(f'Attempting to call a slot on an object {obj_id} which is not found on this loop')
                self.receivers[emitter].remove(obj_id)
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

            self.curr_emitter = emitter
            slot_callable(*args)

    def _calculate_timeout(self) -> Timer:
        # This can potentially be replaced with a sorted set of timers to optimize this linear search for the
        # closest timer.
        closest_timer = min(self.timers, key=lambda t: t.next_timeout())
        return closest_timer

    def exec(self) -> StatusCode:
        status: StatusCode = EventLoopStatus.NORMAL_TERMINATION

        self.default_timer.start()  # this will add timer to the loop's list of timers

        try:
            while not self.should_terminate:
                closest_timer = self._calculate_timeout()

                try:
                    # loop over all incoming signals, see if any of the objects living on this event loop are connected
                    # to this particular signal, call slots if needed
                    signals = self.signal_queue.get_many(timeout=closest_timer.remaining_time())
                except Empty:
                    signals = ()
                finally:
                    if closest_timer.remaining_time() <= 0:
                        # this is inefficient if we have a lot of short timers, but should do for now
                        for t in self.timers:
                            if t.remaining_time() <= 0:
                                t.timeout()

                for signal in signals:
                    self._process_signal(signal)
        except Exception as exc:
            log.warning(f'Unhandled exception {exc} in evt loop {self.object_id}')
            raise exc
        except KeyboardInterrupt:
            log.info(f'Keyboard interrupt detected in the event loop {self}, exiting...')
            status = EventLoopStatus.INTERRUPTED

        return status

    def __str__(self):
        return f'EvtLoop {process_name(self.process)}'


class Timer(EventLoopObject):
    def __init__(self, event_loop: EventLoop, interval_sec: float, single_shot=False, object_id=None):
        super().__init__(event_loop, object_id)

        self._interval_sec = interval_sec
        self._single_shot = single_shot
        self._is_active = False
        self._next_timeout = None
        self.stop()

    def set_interval(self, interval_sec: float):
        self._interval_sec = interval_sec
        if self._is_active:
            self._next_timeout = min(self._next_timeout, time.time() + self._interval_sec)

    def stop(self):
        if self._is_active:
            self._is_active = False
            self.event_loop.remove_timer(self)

        self._next_timeout = time.time() + 1e10

    def start(self):
        if not self._is_active:
            self._is_active = True
            self.event_loop.add_timer(self)

        self._next_timeout = time.time() + self._interval_sec

    def timeout(self):
        self.emit('timeout')
        if self._single_shot:
            self.stop()
        else:
            self._next_timeout += self._interval_sec

    def next_timeout(self) -> float:
        return self._next_timeout

    def remaining_time(self) -> float:
        return max(0, self._next_timeout - time.time())

    def _default_obj_id(self):
        return f'{Timer.__name__}_{super()._default_obj_id()}'


class EventLoopProcess(multiprocessing.Process, EventLoopObject):
    def __init__(self, unique_process_name, daemon=None):
        multiprocessing.Process.__init__(self, target=self._target, name=unique_process_name, daemon=daemon)
        self.event_loop = EventLoop(f'{self.name}_evt_loop')
        EventLoopObject.__init__(self, self.event_loop, self.name)

    def _target(self):
        self.event_loop.exec()

    def start(self):
        self.event_loop.process = self
        super().start()

    def terminate(self) -> None:
        self.event_loop.terminate()


def process_name(p: Union[psutil.Process, multiprocessing.Process]):
    if isinstance(p, psutil.Process):
        return p.name()
    elif isinstance(p, multiprocessing.Process):
        return p.name
    else:
        raise RuntimeError(f'Unknown process type {type(p)}')
