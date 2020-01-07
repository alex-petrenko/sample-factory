import logging
import multiprocessing
import sys
import time

import ray
import torch
from torch.multiprocessing import Process as TorchProcess
from torch.multiprocessing import Queue as TorchQueue

import ray.pyarrow_files.pyarrow as pa
from ray.pyarrow_files.pyarrow import plasma


q = multiprocessing.Queue()
tq = TorchQueue()


def sample_data(single_tensor=False):
    if single_tensor:
        return torch.rand([1000, 128, 72, 3], dtype=torch.float)
    else:
        data = []
        for i in range(1000):
            t = torch.rand([128, 72, 3], dtype=torch.float)
            data.append(t)

        return data


def plasma_shared_mem_process(plasma_store_name):
    plasma_client = plasma.connect(plasma_store_name)
    serialization_context = pa.default_serialization_context()

    counter = 0

    while True:
        data = q.get()
        counter += 1

        if data is None:
            return

        data = plasma_client.get(data, -1, serialization_context=serialization_context)

        print('Received data:', len(data), data, counter)


def test_plasma_memshare():
    ray.init(
        local_mode=False,
        memory=int(1e10), object_store_memory=int(1e10),
        redis_max_memory=int(1e9), driver_object_store_memory=int(1e9),
        logging_level=logging.CRITICAL,
    )

    global_worker = ray.worker.global_worker
    plasma_store_name = global_worker.node.plasma_store_socket_name

    plasma_client = plasma.connect(plasma_store_name)
    serialization_context = pa.default_serialization_context()

    p = multiprocessing.Process(target=plasma_shared_mem_process, args=(plasma_store_name, ))
    p.start()

    start = time.time()

    n = 50
    for i in range(n):
        data = sample_data()

        for j in range(len(data)):
            data[j] = data[j].numpy()
        data = plasma_client.put(
            data, None, serialization_context=serialization_context,
        )

        q.put(data)

        print(f'Progress {i}/{n}')

    q.put(None)
    p.join()

    print(f'Finished sending {n} tensor lists!')
    took_seconds = time.time() - start

    ray.shutdown()

    return took_seconds


# tq = TorchQueue()


def torch_shared_mem_process():
    counter = 0

    while True:
        data = tq.get()
        counter += 1

        if data is None:
            return

        print('Received data:', len(data), data, counter)


def test_mem_share(share_memory):
    p = TorchProcess(target=torch_shared_mem_process)
    p.start()

    start = time.time()

    n = 50
    for i in range(n):
        data = sample_data()

        for data_item in data:
            if share_memory:
                data_item.share_memory_()

        tq.put(data)

        print(f'Progress {i}/{n}')

    tq.put(None)
    p.join()

    print(f'Finished sending {n} tensor lists!')

    took_seconds = time.time() - start
    return took_seconds


def main():
    plasma_memshare = test_plasma_memshare()
    no_shared_memory = test_mem_share(share_memory=False)
    with_shared_memory = test_mem_share(share_memory=True)
    print(f'Took {plasma_memshare:.1f} s plasma shared memory.')
    print(f'Took {no_shared_memory:.1f} s without shared memory.')
    print(f'Took {with_shared_memory:.1f} s with shared memory.')


if __name__ == '__main__':
    sys.exit(main())
