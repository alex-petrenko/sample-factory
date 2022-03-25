import datetime
from unittest import TestCase

from sample_factory.signal_slot.signal_slot import EventLoopObject, process_name, EventLoop, EventLoopProcess, Timer, \
    connect
from sample_factory.utils.utils import log


class C1(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.x = 0

    def inc(self, data):
        self.x += 1
        log.debug(f'inc slot {self.object_id} {self.x=} {data=} process={process_name(self.event_loop.process)}')
        self.emit('reply', process_name(self.event_loop.process), self.x)

    def on_timeout(self):
        log.debug(f'on_timeout slot {self.object_id} {self.x=} process={process_name(self.event_loop.process)}, {datetime.datetime.now()}')


class C2(EventLoopObject):
    def __init__(self, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.pi = 3.14

    def foo(self, data):
        log.debug('Foo')
        self.emit('foo_signal', data)

    def on_reply(self, p, x):
        log.debug('reply from %s(%d) received by %s %s', p, x, self.object_id, process_name(self.event_loop.process))


class TestUtils(TestCase):
    def test_basic(self):
        # create some objects, connect them, run the event loop
        event_loop = EventLoop('main_loop')

        o1 = C1(event_loop, 'o1')
        o2 = C2(event_loop, 'o2')

        o2.connect('foo_signal', o1.inc)
        o2.disconnect('foo_signal', o1.inc)
        o2.connect('foo_signal', o1.inc)

        p = EventLoopProcess(unique_process_name='my_process1')
        o3 = C1(p.event_loop, 'o3_p')
        o4 = C2(p.event_loop, 'o4_p')

        t = Timer(p.event_loop, 2.0)
        t.start()

        connect(o2, 'foo_signal', o3, 'inc')
        connect(o4, 'foo_signal', o3, 'inc')
        connect(o4, 'foo_signal', o3, 'inc')

        p2 = EventLoopProcess(unique_process_name='my_process2')
        o5 = C1(p2.event_loop, 'o5_p2')
        o6 = C2(p2.event_loop, 'o6_p2')

        connect(o2, 'foo_signal', o5, 'inc')
        connect(o6, 'foo_signal', o5, 'inc')
        connect(o6, 'foo_signal', o5, 'inc')

        connect(o5, 'reply', o2, 'on_reply')
        connect(o5, 'reply', o4, 'on_reply')
        connect(o5, 'reply', o6, 'on_reply')
        connect(o5, 'reply', o6, 'on_reply')

        o6.detach()
        del o6

        o7 = C1(p2.event_loop, 'o7_p2')
        o8 = C2(p2.event_loop, 'o8_p2')
        connect(o5, 'reply', o8, 'on_reply')

        connect(t, 'timeout', o7, 'on_timeout')
        connect(t, 'timeout', o1, 'on_timeout')

        terminate_timer = Timer(event_loop, 6.1, single_shot=True)
        terminate_timer.connect('timeout', event_loop, 'terminate')
        terminate_timer.connect('timeout', p, 'terminate')
        terminate_timer.connect('timeout', p2, 'terminate')
        terminate_timer.start()

        p.start()
        p2.start()

        o2.foo(123)

        event_loop.exec()

        p.join()
        p2.join()
