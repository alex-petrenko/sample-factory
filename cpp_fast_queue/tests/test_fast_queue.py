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
            pass
        except Exception as exc:
            log.exception(exc)

    log.info('Done! %d', p_idx)


def consume(q, p_idx, consume_many, total_num_messages=int(1e9)):
    messages_received = 0

    while True:
        try:
            msgs = q.get_many(timeout=0.01, max_messages_to_get=consume_many)

            for msg in msgs:
                messages_received += 1
                if msg[0] % 10000 == 0:
                    log.info('Consume: %r %d num_msgs: %d', msg, p_idx, len(msgs))

            if messages_received >= total_num_messages:
                break
        except Empty:
            if q.is_closed():
                break
        except Exception as exc:
            log.exception(exc)

    log.info('Done! %d', p_idx)


class TestFastQueue(TestCase):
    def test_singleproc(self):
        q = Queue()
        produce(q, 0, num_messages=20)
        consume(q, 0, consume_many=2, total_num_messages=20)
        q.close()

    def test_multiproc(self):
        q = Queue()

        consume_many = 10

        producers = []
        consumers = []
        for j in range(1):
            p = multiprocessing.Process(target=produce, args=(q, j, 500001))
            producers.append(p)
        for j in range(1):
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
        self.assertEqual(py_obj, res)

# fast queue 200K messages per producer
# 20p 3c consume_many = 1: 29.9s
# 20p 3c consume_many = 2: 17.2s
# 20p 3c consume_many = 10: 4.7s
# 20p 3c consume_many = 100: 2.4s
# 20p 3c consume_many = 1000: 2.3s

# 3p 20c consume_many = 1: 4.52s
# 3p 20c consume_many = 2: 4.30s
# 3p 20c consume_many = 10: 4.27s
# 3p 20c consume_many = 100: 4.37s
# 3p 20c consume_many = 1000: 4.29s

# fast queue 500K messages per producer
# 1p 1c consume_many = 1: 2.11s
# 1p 1c consume_many = 2: 2.11s
# 1p 1c consume_many = 10: 2.21s

# multiprocessing.Queue
# 20p 3c: 64s
# 3p 20c: 18s
# 1p 1c: 6.4s
