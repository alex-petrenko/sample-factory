import sys
import time

import ray

from algorithms.appo.env_runner import VectorEnvRunner, TaskType
from algorithms.appo.tests.test_env_runner_ray import make_env_singleplayer
from algorithms.utils.algo_utils import num_env_steps
from algorithms.utils.multi_env import safe_get
from utils.timing import Timing
from utils.utils import log


@ray.remote(num_cpus=0.33)
class MultiprocessNoSync:
    """
    Benchmark (@ray.remote CPU not specified):
    1 worker: 3321 FPS
    4 workers: 12498 FPS
    8 workers: 22212 FPS
    16 workers: 36596 FPS
    20 workers: 38092 FPS (same as number of CPU cores)
    32 workers: 35291 FPS (worse)
    64 workers: 32577 FPS (worse)

    Benchmark (@ray.remote CPU=0.5):
    4 workers: 12117 FPS
    8 workers: 22129 FPS
    16 workers: 37003 FPS
    20 workers: 38339 FPS
    32 workers: 48521 FPS
    64 workers: failed (not enough resources to start actors)

    Benchmark (@ray.remote CPU=0.33):
    4 workers: 12783 FPS
    8 workers: 21821 FPS
    16 workers: 37872 FPS
    20 workers: 39004 FPS
    26 workers: 44472 FPS
    30 workers: 47977 FPS
    32 workers: 49343 FPS
    36 workers: 50067 FPS
    40 workers: 48342 FPS
    48 workers: 47595 FPS (worse)
    64 workers: 47714 FPS (worse)

    Benchmark (@ray.remote CPU=0.33, rendering 160x120):
    36 workers: 68000 FPS
    """

    def __init__(self, make_env_func, frames_to_collect, worker_id):
        self.worker_id = worker_id
        self.frames_to_collect = frames_to_collect
        self.env = make_env_func(None)
        self.action_space = self.env.action_space

    def reset(self):
        self.env.reset()
        log.info('Initialized worker %d', self.worker_id)
        return None

    def step(self):
        num_frames = 0
        while num_frames < self.frames_to_collect:
            actions = self.action_space.sample()
            _, _, done, info = self.env.step(actions)
            if done:
                self.env.reset()

            num_frames += num_env_steps([info])

        return None

    def close(self):
        self.env.close()
        log.info('Finished worker %d', self.worker_id)
        return None


def multiprocess_no_sync():
    total_frames = 200000
    num_workers = 36
    frames_per_worker = total_frames / num_workers
    workers = [MultiprocessNoSync.remote(make_env_singleplayer, frames_per_worker, i) for i in range(num_workers)]

    max_parallel_init = 10
    for i in range(0, num_workers, max_parallel_init):
        reset_tasks = [w.reset.remote() for w in workers[i:i + max_parallel_init]]
        _ = [ray.get(t) for t in reset_tasks]

    timing = Timing()
    with timing.timeit('step'):
        step_tasks = [w.step.remote() for w in workers]
        _ = [ray.get(t) for t in step_tasks]
    log.info('Finished in %.3f seconds, fps %.1f', timing.step, total_frames / timing.step)

    close_tasks = [w.close.remote() for w in workers]
    _ = [ray.get(t) for t in close_tasks]


@ray.remote(num_cpus=0.3)
class VectorRunnerNoSync:
    """
    Benchmark (@ray.remote CPU not specified):
    With vector=2
    1 worker: 2279 FPS
    With vector=10
    8 workers: 17112 FPS
    16 workers: 28243 FPS
    20 workers: 29800 FPS
    32 workers: 27834 FPS

    Benchmark (@ray.remote CPU=0.3):
    With vector=10
    8 workers: 17028 FPS
    16 workers: 27906 FPS
    20 workers: 29368 FPS or 37264 FPS with MT
    32 workers: 35689 FPS or 42453 FPS with MT
    36 workers: 37702 FPS or 44698 FPS with MT
    40 workers: 39128 FPS or 47047 FPS with MT
    48 workers: ----- FPS or 49916 FPS with MT
    60 workers: ----- FPS or 48650 FPS with MT

    With vector=30
    20 workers: 38119 FPS
    """

    def __init__(self, frames_to_collect, worker_id):
        self.worker_id = worker_id
        self.num_frames = 0
        self.frames_to_collect = frames_to_collect

        self.action_space = None
        self.env_runner = None

    def sync(self):
        return None

    def init(self, make_env_func, vector_size, num_splits):
        tmp_env = make_env_func(None)
        self.action_space = tmp_env.action_space
        tmp_env.close()
        del tmp_env

        self.env_runner = VectorEnvRunner(
            make_env_func, vector_size=vector_size, num_splits=num_splits, worker_idx=self.worker_id,
        )
        self.env_runner.init()

        self.env_runner.reset()
        log.info('Initialized worker %d', self.worker_id)
        return None

    def step(self):
        num_frames = 0

        actions = [self.action_space.sample()] * (self.env_runner.vector_size // self.env_runner.num_splits)
        for split_idx in range(self.env_runner.num_splits):
            self.env_runner.task_queue.put((TaskType.STEP, split_idx, actions))

        while num_frames < self.frames_to_collect:
            split_idx, results = safe_get(self.env_runner.result_queue)
            self.env_runner.result_queue.task_done()

            infos = [r[-1] for r in results]

            num_frames += num_env_steps(infos)

            actions = [self.action_space.sample()] * (self.env_runner.vector_size // self.env_runner.num_splits)
            self.env_runner.task_queue.put((TaskType.STEP, split_idx, actions))

        return None

    def close(self):
        self.env_runner.close()
        log.info('Finished worker %d', self.worker_id)
        return None


def vector_runner_no_sync():
    total_frames = 200000
    num_workers = 20
    vector_size = 30
    num_splits = 2
    frames_per_worker = total_frames / num_workers

    workers = []
    for i in range(num_workers):
        worker = VectorRunnerNoSync.remote(frames_per_worker, i)
        _ = ray.get(worker.sync.remote())
        workers.append(worker)

    max_parallel_init = 10
    for i in range(0, num_workers, max_parallel_init):
        reset_tasks = [
            w.init.remote(make_env_singleplayer, vector_size, num_splits)
            for w in workers[i:i + max_parallel_init]
        ]
        _ = [ray.get(t) for t in reset_tasks]

    timing = Timing()
    with timing.timeit('step'):
        step_tasks = [w.step.remote() for w in workers]
        _ = [ray.get(t) for t in step_tasks]
    log.info('Finished in %.3f seconds, fps %.1f', timing.step, total_frames / timing.step)

    close_tasks = [w.close.remote() for w in workers]
    _ = [ray.get(t) for t in close_tasks]


def main():
    """Script entry point."""
    ray.init(local_mode=False)

    # multiprocess_no_sync()
    vector_runner_no_sync()

    time.sleep(0.5)
    ray.shutdown()
    log.info('Done!')


if __name__ == '__main__':
    sys.exit(main())
