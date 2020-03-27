import copy
import sys
import time

import numpy as np
import torch

use_torch_multiprocessing = True

if use_torch_multiprocessing:
    from torch.multiprocessing import Process as TorchProcess, Event
    from torch.multiprocessing import JoinableQueue as TorchQueue
else:
    from multiprocessing import Process as TorchProcess, Event
    from multiprocessing import JoinableQueue as TorchQueue

from utils.timing import Timing
from utils.utils import log


q = TorchQueue()


def worker(evt):
    counter = 0
    a = []
    timing = Timing()

    tensor = None
    is_ready_tensor = None

    while True:
        log.debug('Reading from queue...')
        with timing.add_time('read'):
            data = q.get()
        log.debug('Done Reading!!')
        counter += 1

        if data is None:
            is_ready_tensor[0] = 1
            break
        elif isinstance(data, torch.Tensor):
            is_ready_tensor = data
            continue
        elif isinstance(data, dict):
            tensor = data['d']['t'][0]

            tensor_copy = copy.deepcopy(tensor)
            print(tensor_copy)

        value = tensor[0][0][0][0].cpu().detach().item()
        a.append(value)
        log.debug('Received data: %.4f, %d', value, counter)
        # tensor.fill_(42.0)
        tensor[0][0][0][0] = 42.0
        log.debug('After tensor fill: %.4f', tensor[0][0][0][0].cpu().detach().item())

        is_ready_tensor[0] = 1

        # data.share_memory_()
        # q.task_done()

        # evt.set()

    log.info('%r', a)
    log.info('Timing: %s', timing)


def main():
    evt = Event()

    p = TorchProcess(target=worker, args=(evt, ))

    timing = Timing()

    with timing.timeit('all'):
        n = 30

        with timing.timeit('data'):
            log.info('Generating data...')
            np.random.seed(0)
            all_data = np.random.random([n, 1000, 128, 72, 3])
            log.info('Done!')

        shared_mem = True
        shared_tensor = None
        if shared_mem:
            shared_tensor = torch.from_numpy(all_data[0]).float()
            shared_tensor.share_memory_()
        tensor_sent = False

        is_ready_tensor = torch.zeros([1], dtype=torch.int32)
        is_ready_tensor.share_memory_()

        p.start()

        q.put(is_ready_tensor)

        with timing.timeit('sending'):
            for i in range(n):
                with timing.add_time('numpy'):
                    t = torch.from_numpy(all_data[i]).float()

                if shared_mem:
                    with timing.add_time('copy'):
                        shared_tensor.copy_(t)
                    with timing.add_time('queue'):
                        is_ready_tensor[0] = 0

                        if tensor_sent:
                            q.put(False)
                        else:
                            dictionary = dict(d=dict(t=[shared_tensor]))
                            q.put(dictionary)
                            tensor_sent = True

                        # log.info('waiting evt...')
                        # evt.wait()
                        # evt.clear()
                        # log.info('Done waiting!')
                        # shared_tensor.fill_(-42.0)


                        while is_ready_tensor[0] == 0:
                            time.sleep(0.001)
                        # q.join()

                        log.info('Data: %.3f', shared_tensor[0, 0, 0, 0].cpu().detach().item())
                else:
                    with timing.add_time('queue'):
                        dictionary = dict(d=dict(t=[t]))
                        q.put(dictionary)
                        q.join()

                log.info('Progress %d/%d', i, n)

        with timing.timeit('fin'):
            is_ready_tensor[0] = 0
            q.put(None)
            while is_ready_tensor[0] == 0:
                time.sleep(0.001)
            p.join()

    log.info('Timing: %s', timing)


if __name__ == '__main__':
    sys.exit(main())
