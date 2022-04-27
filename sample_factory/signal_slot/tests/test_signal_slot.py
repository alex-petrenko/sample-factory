import datetime
import multiprocessing
from unittest import TestCase

from sample_factory.signal_slot.signal_slot import EventLoopObject, process_name, EventLoop, EventLoopProcess, Timer, \
    signal
from sample_factory.utils.utils import log


class C1(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.x = 0

    @signal
    def reply(self): pass

    @signal
    def broadcast_signal(self): pass

    def inc(self, data):
        self.x += 1
        log.debug(f'inc slot {self.object_id} {self.x=} {data=} process={process_name(self.event_loop.process)}')
        self.reply.emit(process_name(self.event_loop.process), self.x)

    def on_timeout(self):
        log.debug(f'on_timeout slot {self.object_id} {self.x=} process={process_name(self.event_loop.process)}, {datetime.datetime.now()}')
        self.broadcast_signal.emit(42, 43)

    def on_bcast(self, arg1, arg2):
        log.info(f'on_bcastC1 slot {self.object_id} {arg1=} {arg2=} process={process_name(self.event_loop.process)}, {datetime.datetime.now()}')


class C2(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.pi = 3.14

    @signal
    def foo_signal(self): pass

    def foo(self, data):
        log.debug('Foo')
        self.foo_signal.emit(data)

    def on_reply(self, p, x):
        log.debug('reply from %s(%d) received by %s %s', p, x, self.object_id, process_name(self.event_loop.process))

    def on_bcast(self, arg1, arg2):
        log.info(f'on_bcastC2 slot {self.object_id} {arg1=} {arg2=} process={process_name(self.event_loop.process)}, {datetime.datetime.now()}')


class TestSignalSlot(TestCase):
    def test_basic(self):
        # create some objects, connect them, run the event loop
        event_loop = EventLoop('main_loop')

        o1 = C1(event_loop, 'o1')
        o2 = C2(event_loop, 'o2')

        o2.foo_signal.connect(o1.inc)
        o2.foo_signal.disconnect(o1.inc)
        o2.foo_signal.connect(o1.inc)

        p = EventLoopProcess(unique_process_name='my_process1')
        o3 = C1(p.event_loop, 'o3_p')
        o4 = C2(p.event_loop, 'o4_p')

        t = Timer(p.event_loop, 2.0)

        o2.foo_signal.connect(o3.inc)
        o4.foo_signal.connect(o3.inc)
        o4.foo_signal.connect(o3.inc)

        p2 = EventLoopProcess(unique_process_name='my_process2')
        o5 = C1(p2.event_loop, 'o5_p2')
        o6 = C2(p2.event_loop, 'o6_p2')

        o2.foo_signal.connect(o5.inc)
        o6.foo_signal.connect(o5.inc)

        o5.reply.connect(o2.on_reply)
        o5.reply.connect(o4.on_reply)
        o5.reply.connect(o6.on_reply)
        o5.reply.connect(o6.on_reply)

        o6.detach()
        del o6

        o7 = C1(p2.event_loop, 'o7_p2')
        o8 = C2(p2.event_loop, 'o8_p2')

        o5.reply.connect(o8.on_reply)

        o1.broadcast_signal.broadcast_on(p2.event_loop)

        o7.subscribe('broadcast_signal', o7.on_bcast)
        o8.subscribe('broadcast_signal', o8.on_bcast)

        t.timeout.connect(o7.on_timeout)
        t.timeout.connect(o1.on_timeout)

        stop_timer = Timer(event_loop, 6.1, single_shot=True)
        stop_timer.timeout.connect(event_loop.stop)
        stop_timer.timeout.connect(p.stop)
        stop_timer.timeout.connect(p2.stop)

        p.start()
        p2.start()

        o2.foo(123)

        event_loop.exec()

        p.join()
        p2.join()

    class C3(EventLoopObject):
        @signal
        def s1(self): pass

    class C4(EventLoopObject):
        def on_start(self):
            log.debug(f'{self.on_start.__name__} start')

        def on_s1(self, arg1, arg2):
            log.info(f'{self.on_s1.__name__} {arg1=} {arg2=}')

    def test_multiarg(self):
        ctx = multiprocessing.get_context('spawn')
        p = EventLoopProcess('_p1', ctx)

        event_loop = EventLoop('multiarg_loop')
        stop_timer = Timer(event_loop, 0.5, single_shot=True)
        stop_timer.timeout.connect(event_loop.stop)

        o1 = self.C3(p.event_loop, 'o1_')
        o2 = self.C4(p.event_loop, 'o2_')

        p.event_loop.start.connect(o2.on_start)

        o1.s1.connect(o2.on_s1)
        o1.s1.emit(dict(a=1, b=2, c=3), 42)
        o1.s1.emit_many([(dict(a=1, b=2, c=3), 42), (dict(a=11, b=22, c=33), 422)])

        event_loop.terminate.connect(p.stop)

        p.start()
        log.debug(f'Starting event loop {event_loop}')
        event_loop.exec()
        p.join()

    def test_process_given_evt_loop(self):
        main_loop = EventLoop('__main_loop')
        p_loop = EventLoop('__p1_loop')

        stop_timer = Timer(main_loop, 0.5, single_shot=True)
        stop_timer.timeout.connect(main_loop.stop)

        o1 = self.C3(p_loop, 'o1_')
        o2 = self.C4(p_loop, 'o2_')

        o1.s1.connect(o2.on_s1)
        o1.s1.emit(dict(a=1, b=2, c=3), 42)

        p = EventLoopProcess('__p1', p_loop)
        main_loop.terminate.connect(p.stop)

        p.start()
        log.debug(f'Starting event loop {main_loop}')
        main_loop.exec()
        p.join()

