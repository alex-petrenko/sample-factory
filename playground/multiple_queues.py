import random
import select
import time
from multiprocessing import JoinableQueue, Process

from algorithms.utils.multi_env import safe_get
from utils.utils import log


class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.result_queue = JoinableQueue()
        self.process = Process(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        for i in range(100):
            time.sleep(random.random())
            self.result_queue.put(self.worker_id ** 2)


def main():
    num_workers = 4
    workers = [Worker(i) for i in range(num_workers)]

    workers_by_handle = dict()
    for w in workers:
        workers_by_handle[w.result_queue._reader._handle] = w

    while True:
        queues = [w.result_queue._reader for w in workers]
        ready, _, _ = select.select(queues, [], [], 0.01)

        # log.info('Queues ready for reading: %d', len(ready))

        for ready_queue in ready:
            r = ready_queue
            w = workers_by_handle[r._handle]
            result = safe_get(w.result_queue)
            w.result_queue.task_done()
            log.info('Worker %d finished, result %d!', w.worker_id, result)


if __name__ == '__main__':
    main()
