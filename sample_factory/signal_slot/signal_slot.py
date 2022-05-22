from __future__ import annotations

import multiprocessing
import os
import time
import types
import uuid
from dataclasses import dataclass
from queue import Empty
from typing import Dict, Any, Set, Callable, Union, List, Optional, Iterable, Tuple

import psutil

from sample_factory.algo.utils.queues import get_queue
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


# noinspection PyPep8Naming
class signal:
    def __init__(self, _):
        self._name = None
        self._obj: Optional[EventLoopObject] = None

    @property
    def obj(self):
        return self._obj

    @property
    def name(self):
        return self._name

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        assert isinstance(obj, EventLoopObject), \
            f'signals can only be added to {EventLoopObject.__name__}, not {type(obj)}'
        self._obj = obj
        return self

    def connect(self, other: Union[EventLoopObject, BoundMethod], slot: str = None):
        self._obj.connect(self._name, other, slot)

    def disconnect(self, other: Union[EventLoopObject, BoundMethod], slot: str = None):
        self._obj.disconnect(self._name, other, slot)

    def emit(self, *args):
        self._obj.emit(self._name, *args)

    def emit_many(self, list_of_args: Iterable[Tuple]):
        self._obj.emit_many(self._name, list_of_args)

    def broadcast_on(self, event_loop: EventLoop):
        self._obj.register_broadcast(self._name, event_loop)


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

        assert isinstance(obj, EventLoopObject), f'slot should be a method of {EventLoopObject.__name__}'
        assert slot is not None
        return obj, slot

    def connect(self, signal_: str, other: EventLoopObject | BoundMethod, slot: str = None):
        other, slot = self._bound_method_to_obj_slot(other, slot)

        self._throw_if_different_processes(self, other)

        emitter = Emitter(self.object_id, signal_)
        receiver_id = other.object_id

        self._add_to_dict_of_sets(self.send_signals_to, signal_, receiver_id)

        receiving_loop = other.event_loop
        self._add_to_dict_of_sets(receiving_loop.receivers, emitter, receiver_id)

        q = receiving_loop.signal_queue
        self.receiver_queues[receiver_id] = q
        self.receiver_refcount[receiver_id] = self.receiver_refcount.get(receiver_id, 0) + 1

        other.connections[emitter] = slot

    def disconnect(self, signal_, other: EventLoopObject | BoundMethod, slot: str = None):
        other, slot = self._bound_method_to_obj_slot(other, slot)

        self._throw_if_different_processes(self, other)

        if signal_ not in self.send_signals_to:
            log.warning(f'{self.object_id}:{signal_=} is not connected to anything')
            return

        receiver_id = other.object_id
        if receiver_id not in self.send_signals_to[signal_]:
            log.warning(f'{self.object_id}:{signal_=} is not connected to {receiver_id}:{slot=}')
            return

        self.send_signals_to[signal_].remove(receiver_id)

        self.receiver_refcount[receiver_id] -= 1
        if self.receiver_refcount[receiver_id] <= 0:
            del self.receiver_refcount[receiver_id]
            del self.receiver_queues[receiver_id]

        emitter = Emitter(self.object_id, signal_)
        del other.connections[emitter]

        loop_receivers = other.event_loop.receivers.get(emitter)
        if loop_receivers is not None:
            loop_receivers.remove(other.object_id)

    def register_broadcast(self, signal_: str, event_loop: EventLoop):
        self.connect(signal_, event_loop.broadcast)

    def subscribe(self, signal_: str, slot: Union[BoundMethod, str]):
        if isinstance(slot, (types.MethodType, types.BuiltinMethodType)):
            slot = slot.__name__
        self.event_loop.connect(signal_, self, slot)

    def unsubscribe(self, signal_: str, slot: Union[BoundMethod, str]):
        if isinstance(slot, (types.MethodType, types.BuiltinMethodType)):
            slot = slot.__name__
        self.event_loop.disconnect(signal_, self, slot)

    def emit(self, signal_: str, *args):
        self.emit_many(signal_, (args, ))

    def emit_many(self, signal_: str, list_of_args: Iterable[Tuple]):
        pid = self.event_loop.process.pid
        if os.getpid() != pid:
            raise RuntimeError(
                f'Cannot emit {signal_}: object {self.object_id} lives on a different process {pid}!'
            )

        signals_to_emit = tuple((self.object_id, signal_, args) for args in list_of_args)

        # find a set of queues we need to send this signal to
        queues = set()
        for receiver_id in self.send_signals_to.get(signal_, ()):
            queues.add(self.receiver_queues[receiver_id])

        for q in queues:
            # we just push messages into each receiver event loop queue
            # event loops themselves will redistribute the signals to all receivers living on that loop
            q.put_many(signals_to_emit, block=True, timeout=0.1)

    def detach(self):
        """Detach the object from it's current event loop."""
        if self.event_loop:
            del self.event_loop.objects[self.object_id]
            self.event_loop = None

    def __del__(self):
        self.detach()
        self._obj_ids.remove(self.object_id)


class EventLoopStatus:
    NORMAL_TERMINATION, INTERRUPTED = range(2)


class EventLoop(EventLoopObject):
    def __init__(self, unique_loop_name, serial_mode=False):
        # objects living on this loop
        self.objects: Dict[ObjectID, EventLoopObject] = dict()

        super().__init__(self, unique_loop_name)

        # object responsible for stopping the loop (if any)
        self.owner: Optional[EventLoopObject] = None

        # when event loop is created it just lives on the current process
        self.process: Union[psutil.Process, EventLoopProcess] = psutil.Process(os.getpid())

        self.signal_queue = get_queue(serial=serial_mode, buffer_size_bytes=5_000_000)

        # Separate container to keep track of timers living on this thread. Start with one default timer.
        self.timers: List[Timer] = []
        self.default_timer = Timer(self, 0.05, object_id=f'{self.object_id}_timer')

        self.receivers: Dict[Emitter, Set[ObjectID]] = dict()

        # emitter of the signal which is currently being processed
        self.curr_emitter: Optional[Emitter] = None

        self.should_terminate = False

        self.verbose = False

        # connect to our own termination signal
        self._internal_terminate.connect(self._terminate)

    """Emitted right before the start of the loop."""
    @signal
    def start(self): pass

    """Emitted upon loop termination."""
    @signal
    def terminate(self): pass

    """Internal signal: do not connect to this."""
    @signal
    def _internal_terminate(self): pass

    def add_timer(self, t: Timer):
        self.timers.append(t)

    def remove_timer(self, t: Timer):
        self.timers.remove(t)

    def stop(self):
        """
        Graceful termination: the loop will process all unprocessed signals before exiting.
        After this the loop does only one last iteration, if any new signals are emitted during this last iteration
        they will be ignored.
        """
        self._internal_terminate.emit()

    def _terminate(self):
        """Forceful termination, some of the signals currently in the queue might remain unprocessed."""
        self.should_terminate = True

    def broadcast(self, *args):
        curr_signal = self.curr_emitter.signal_name

        # we could re-emit the signal to reuse the existing signal propagation mechanism, but we can avoid
        # doing this to reduce overhead
        self._process_signal((self.object_id, curr_signal, args))

    def _process_signal(self, signal_):
        if self.verbose:
            log.debug(f'{self} received {signal_=}...')

        emitter_object_id, signal_name, args = signal_
        emitter = Emitter(emitter_object_id, signal_name)

        receiver_ids = tuple(self.receivers.get(emitter, ()))

        for obj_id in receiver_ids:
            obj = self.objects.get(obj_id)
            if obj is None:
                if self.verbose:
                    log.warning(f'{self} attempting to call a slot on an object {obj_id} which is not found on this loop ({signal_=})')
                self.receivers[emitter].remove(obj_id)
                continue

            slot = obj.connections.get(emitter)
            if obj is None:
                log.warning(f'{self} {emitter=} does not appear to be connected to {obj_id=}')
                continue

            if not hasattr(obj, slot):
                log.warning(f'{self} {slot=} not found in object {obj_id}')
                continue

            slot_callable = getattr(obj, slot)
            if not isinstance(slot_callable, Callable):
                log.warning(f'{self} {slot=} of {obj_id=} is not callable')
                continue

            self.curr_emitter = emitter
            if self.verbose:
                log.debug(f'{self} calling slot {obj_id}:{slot}')

            # noinspection PyBroadException
            try:
                slot_callable(*args)
            except Exception as exc:
                log.exception(f'{self} unhandled exception in {slot=} connected to {emitter=}')
                raise exc

    def _calculate_timeout(self) -> Timer:
        # This can potentially be replaced with a sorted set of timers to optimize this linear search for the
        # closest timer.
        closest_timer = min(self.timers, key=lambda t: t.next_timeout())
        return closest_timer

    def exec(self) -> StatusCode:
        status: StatusCode = EventLoopStatus.NORMAL_TERMINATION

        self.default_timer.start()  # this will add timer to the loop's list of timers

        try:
            self.start.emit()

            while True:
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
                                t.fire()

                for s in signals:
                    self._process_signal(s)

                if self.should_terminate:
                    log.debug(f'Loop {self.object_id} terminating...')
                    self.terminate.emit()
                    break

        except Exception as exc:
            log.warning(f'Unhandled exception {exc} in evt loop {self.object_id}')
            raise exc
        except KeyboardInterrupt:
            log.info(f'Keyboard interrupt detected in the event loop {self}, exiting...')
            status = EventLoopStatus.INTERRUPTED

        return status

    def __str__(self):
        return f'EvtLoop [{self.object_id}, process={process_name(self.process)}]'


class Timer(EventLoopObject):
    def __init__(self, event_loop: EventLoop, interval_sec: float, single_shot=False, object_id=None):
        super().__init__(event_loop, object_id)

        self._interval_sec = interval_sec
        self._single_shot = single_shot
        self._is_active = False
        self._next_timeout = None
        self.start()

    @signal
    def timeout(self): pass

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

    def _emit(self):
        self.timeout.emit()

    def fire(self):
        self._emit()
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


class TightLoop(Timer):
    def __init__(self, event_loop: EventLoop, object_id=None):
        super().__init__(event_loop, 0.0, object_id)

    @signal
    def iteration(self): pass

    def _emit(self):
        self.iteration.emit()


class EventLoopProcess(EventLoopObject):
    def __init__(self, unique_process_name, multiprocessing_context=None, init_func=None, args=(), kwargs=None, daemon=None):
        """
        Here we could've inherited from Process, but the actual class of process (i.e. Process vs SpawnProcess)
        depends on the multiprocessing context and hence is not known during the generation of the class.

        Instead of using inheritance we just wrap a process instance.
        """
        process_cls = multiprocessing.Process if multiprocessing_context is None else multiprocessing_context.Process

        self._process = process_cls(target=self._target, name=unique_process_name, daemon=daemon)

        self._init_func: Optional[Callable] = init_func
        self._args = self._kwargs = None
        self.set_init_func_args(args, kwargs)

        self.event_loop = EventLoop(f'{unique_process_name}_evt_loop')
        EventLoopObject.__init__(self, self.event_loop, unique_process_name)

    def set_init_func_args(self, args=(), kwargs=None):
        assert not self._process.is_alive()
        self._args = tuple(args)
        self._kwargs = dict() if kwargs is None else dict(kwargs)

    def _target(self):
        if self._init_func:
            self._init_func(*self._args, **self._kwargs)
        self.event_loop.exec()

    def start(self):
        self.event_loop.process = self
        self._process.start()

    def stop(self):
        self.event_loop.stop()

    def terminate(self):
        self._process.terminate()

    def kill(self):
        self._process.kill()

    def join(self, timeout=None):
        self._process.join(timeout)

    def is_alive(self):
        return self._process.is_alive()

    def close(self):
        return self._process.close()

    @property
    def name(self):
        return self._process.name

    @property
    def daemon(self):
        return self._process.daemon

    @property
    def exitcode(self):
        return self._process.exitcode

    @property
    def ident(self):
        return self._process.ident

    pid = ident


def process_name(p: Union[psutil.Process, EventLoopProcess]):
    if isinstance(p, psutil.Process):
        return p.name()
    elif isinstance(p, EventLoopProcess):
        return p.name
    else:
        raise RuntimeError(f'Unknown process type {type(p)}')
