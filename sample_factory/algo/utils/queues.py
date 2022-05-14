import os
import time
from queue import Queue, Empty, Full


def get_queue(serial=False, buffer_size_bytes=1_000_000):
    if serial:
        # for serial execution we don't need the multiprocessing queue
        return QueueWrapper()
    else:
        return get_mp_queue(buffer_size_bytes)


def get_mp_queue(buffer_size_bytes=1_000_000):
    if os.name == 'nt':
        from sample_factory.utils.faster_fifo_stub import MpQueueWrapper as MpQueue
    else:
        from faster_fifo import Queue as MpQueue
        # noinspection PyUnresolvedReferences
        import faster_fifo_reduction

    return MpQueue(buffer_size_bytes)


class QueueWrapper(Queue):
    def get_many(self, block=True, timeout=float(1e3), max_messages_to_get=int(1e9)):
        msgs = []

        while len(msgs) < max_messages_to_get:
            try:
                if len(msgs) == 0:
                    msg = self.get(block, timeout)
                else:
                    msg = self.get_nowait()

                msgs.append(msg)
            except Empty:
                break

        if not msgs:
            raise Empty
        return msgs

    def get_many_nowait(self, max_messages_to_get=int(1e9)):
        return self.get_many(block=False, max_messages_to_get=max_messages_to_get)

    def put_many(self, xs, block=True, timeout=float(1e3)):
        started = time.time()

        for x in xs:
            self.put(x, block, timeout)
            time_elapsed = time.time() - started
            timeout = max(0.0, timeout - time_elapsed)

    def put_many_nowait(self, xs):
        self.put_many(xs, block=False)
