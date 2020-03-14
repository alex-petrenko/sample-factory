import random
import sys

import torch

from algorithms.appo.appo_utils import TensorBatcher, ObjectPool
from utils.timing import Timing
from utils.utils import log


def main():
    with torch.no_grad():
        big_tensor = torch.zeros([500, 32, 3, 72, 128], dtype=torch.uint8).random_(0, 255)
        big_tensor.share_memory_()

        timing = Timing()

        prealloc = torch.zeros([2048, 3, 72, 128], dtype=torch.uint8)

        pool = ObjectPool()
        batcher = TensorBatcher(pool)

        repeat = 50

        with timing.timeit('all'):
            for i in range(repeat):
                arr = []
                with timing.add_time('arr'):
                    for _ in range(2048 // 32):
                        idx = random.randint(0, 499)
                        arr.append(big_tensor[idx])

                with timing.add_time('cat'):
                    c = torch.cat(arr)
                log.debug('%d', c.nelement())

                with timing.add_time('prealloc'):
                    offset = 0
                    for a in arr:
                        first_dim = a.shape[0]
                        prealloc[offset:offset+first_dim].copy_(a)
                log.debug('%d', prealloc.nelement())

                with timing.add_time('batcher'):
                    res = batcher.cat(dict(obs=arr), timing)
                log.debug('%d', res['obs'].nelement())
                if random.random() < 0.9:
                    pool.put(res)

        tm = timing.cat / repeat
        tm2 = timing.prealloc / repeat
        tm3 = timing.batcher / repeat

        log.debug('Timing: %s, %.1f, %.1f, %.1f', timing, tm * 244, tm2 * 244, tm3 * 244)


if __name__ == '__main__':
    sys.exit(main())
