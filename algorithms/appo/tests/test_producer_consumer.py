import multiprocessing
import random

import sys
import time
from queue import Empty, Full

from utils.timing import Timing
from utils.utils import log

from multiprocessing.sharedctypes import RawArray, RawValue
from ctypes import Structure, c_int32

N_PRODUCERS = 20
N_CONSUMERS = 3

USE_QUEUE = True

MESSAGES_PER_PRODUCER = int(1e5) + 1
MSG_SIZE = 5


class Message(Structure):
    _fields_ = [('a', c_int32), ('b', c_int32), ('c', c_int32), ('d', c_int32), ('e', c_int32)]

    def to_bytes(self):
        return bytes(self)

    def from_bytes(self, b):
        return self.from_buffer_copy(b)


class LockingRingBuffer:
    def __init__(self):
        self.max_size = 1000000
        self.data = RawArray(Message, self.max_size)
        self.mutex = multiprocessing.Lock()
        self.head_idx = RawValue(c_int32, 0)
        self.tail_idx = RawValue(c_int32, 0)
        self.size = RawValue(c_int32, 0)

    def put(self, msg):
        with self.mutex:
            if self.size.value >= self.max_size:
                raise Full

            self.data[self.tail_idx.value] = msg
            self.tail_idx.value = (self.tail_idx.value + 1) % self.max_size
            self.size.value += 1

    def get(self, n=100):
        with self.mutex:
            size = self.size.value
            if size <= 0:
                raise Empty

            msgs = []
            i = self.head_idx.value

            while True:
                msgs.append(self.data[i])
                i = (i + 1) % self.max_size
                size -= 1
                if len(msgs) >= n or size <= 0:
                    break

            self.head_idx.value = i
            self.size.value = size

            return msgs


class Producer:
    def __init__(self, idx, q, b):
        self.idx = idx
        self.q = q
        self.b = b
        self.proc = multiprocessing.Process(target=self.run)
        self.proc.start()

    def run(self):
        n_messages = MESSAGES_PER_PRODUCER
        msg_size = MSG_SIZE

        for msg_idx in range(n_messages):
            if USE_QUEUE:
                msg = (msg_idx, ) * msg_size
                self.q.put(msg)
            else:
                msg = Message(msg_idx, msg_idx, msg_idx, msg_idx, msg_idx)
                while True:
                    try:
                        self.b.put(msg)
                        break
                    except Full:
                        time.sleep(0.001)
                        continue

            if msg_idx % 1000 == 0:
                log.debug('Producer %d progress %d/%d', self.idx, msg_idx, n_messages)

        log.debug('Producer %d exits', self.idx)

    def stop(self):
        self.proc.join()


class Consumer:
    def __init__(self, idx, q, b):
        self.idx = idx
        self.q = q
        self.b = b
        self.proc = multiprocessing.Process(target=self.run)
        self.proc.start()

    def run(self):
        if USE_QUEUE:
            while True:
                try:
                    msg = self.q.get(timeout=1.0)
                except Empty:
                    break
                else:
                    if msg[0] % 10000 == 0:
                        log.debug('Consumer %d progress %d', self.idx, msg[0])
        else:
            while True:
                try:
                    msgs = self.b.get(n=20)

                    for msg in msgs:
                        if msg.a % 1000 == 0:
                            log.debug('Consumer %d progress %d, num %d', self.idx, msg.a, len(msgs))
                except Empty:
                    time.sleep(0.001)
                    continue

        log.debug('Consumer %d exits', self.idx)

    def stop(self):
        self.proc.join()


def main():
    q = multiprocessing.Queue(maxsize=10000)
    b = LockingRingBuffer()

    producers = []
    for i in range(N_PRODUCERS):
        p = Producer(i, q, b)
        producers.append(p)

    consumers = []
    for i in range(N_CONSUMERS):
        c = Consumer(i, q, b)
        consumers.append(c)

    timing = Timing()

    # with timing.timeit('loop'):
    #     for i in range(int(2e6)):
    #         msg = (0,) * 10
    #         if i % 10000 == 0:
    #             log.debug('%r', msg)

    log.debug('Timing %s', timing)

    with timing.timeit('work'):
        for c in consumers:
            c.stop()

    log.debug('Consumers joined!')

    while q.qsize() > 0:
        q.get()

    for p in producers:
        p.stop()
    log.debug('Producers joined!')

    log.info('Timing %s', timing)

    return 0


if __name__ == '__main__':
    sys.exit(main())
