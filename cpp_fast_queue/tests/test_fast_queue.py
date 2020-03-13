import multiprocessing
from queue import Full, Empty
from unittest import TestCase

from fast_queue import Queue
from utils.utils import log

MSG_SIZE = 5


def make_msg(msg_idx):
    return (msg_idx, ) * MSG_SIZE


def produce(q, p_idx, num_messages):
    i = 0
    while i < num_messages:
        try:
            q.put(make_msg(i), timeout=0.01)

            if i % 1000 == 0:
                log.info('Produce: %d %d', i, p_idx)
            i += 1
        except Full:
            # time.sleep(0.001)
            pass
        except Exception as exc:
            log.exception(exc)

    log.info('Done! %d', p_idx)


def consume(q, p_idx, consume_many):
    while True:
        try:
            msgs = q.get_many(consume_many, timeout=0.01)

            for msg in msgs:
                if msg[0] % 1000 == 0:
                    log.info('Consume: %r %d num_msgs: %d', msg, p_idx, len(msgs))
        except Empty:
            if q.is_closed():
                break
            # time.sleep(0.001)
        except Exception as exc:
            log.exception(exc)

    log.info('Done! %d', p_idx)


class TestFastQueue(TestCase):
    def test_singleproc(self):
        q = Queue()
        produce(q, 0, num_messages=20)
        consume(q, 0, consume_many=2)

    def test_multiproc(self):
        q = Queue()

        consume_many = 2000

        producers = []
        consumers = []
        for j in range(20):
            p = multiprocessing.Process(target=produce, args=(q, j, 1000001))
            producers.append(p)
        for j in range(3):
            p = multiprocessing.Process(target=consume, args=(q, j, consume_many))
            consumers.append(p)

        for c in consumers:
            c.start()
        for p in producers:
            p.start()

        for p in producers:
            p.join()

        q.close()

        for c in consumers:
            c.join()

        log.info('Exit...')

    def test_msg(self):
        q = Queue(max_size_bytes=1000)
        py_obj = dict(a=42, b=33, c=(1, 2, 3), d=[1, 2, 3], e='123', f=b'kkk')
        q.put_nowait(py_obj)

        res = q.get_nowait()
        log.debug('Got object %r', res)
