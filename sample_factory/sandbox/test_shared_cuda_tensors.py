import multiprocessing
import sys
import time

import torch

from sample_factory.algo.utils.multiprocessing_utils import get_mp_queue


def worker1(q):
    print('starting worker1')
    t = torch.randn(1, 2)
    t = t.to(torch.device('cuda'))
    t.share_memory_()
    q.put(t)
    time.sleep(5)
    print('finishing worker1')


def worker2(q: multiprocessing.Queue):
    print('starting worker2')
    t = q.get(block=True, timeout=10.0)
    print(t)
    del t
    time.sleep(10)
    print('finishing worker2')


def main():
    q = get_mp_queue()
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=worker1, args=(q,))
    p.start()

    p2 = ctx.Process(target=worker2, args=(q,))
    p2.start()

    p2.join()


if __name__ == '__main__':
    sys.exit(main())
